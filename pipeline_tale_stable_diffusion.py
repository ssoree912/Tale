from typing import Callable, List, Optional, Tuple, Union, Dict
from torchvision import transforms
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
# from modeling_t5_exceptional import *

class TALEStableDiffusionPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae,
            text_encoder,
            tokenizer,
            unet,
            scheduler,
            safety_checker,
            feature_extractor,
            image_encoder,
            requires_safety_checker,
        )

    @torch.no_grad()
    def img_to_latents(
        self,
        x: torch.FloatTensor, 
    ) -> torch.FloatTensor:
        x = x.to(self.vae.device, self.vae.dtype)
        x = 2. * x - 1.
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.mean * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def latents_to_img(
        self, 
        x: torch.FloatTensor, 
    ):
        x = x.to(self.vae.device)
        image = self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil")
        return image

    # Calculate statistics
    def calc_mean_std(
        self,
        feat: torch.FloatTensor, 
        eps: float = 1e-5
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        # assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return (feat_mean, feat_std)

    # Adaptive Latent Normalization
    def adaptive_normalization(
        self,
        content_feat_ori: torch.FloatTensor, 
        style_feat: torch.FloatTensor, 
        roi: Optional[List[int]] = None,  
        alpha: float = 1., 
        seg: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:

        assert (content_feat_ori.size()[:2] == style_feat.size()[:2])

        if roi is not None:
            content_feat = content_feat_ori[:, :, roi[0]:roi[1], roi[2]:roi[3]].clone()
            size = content_feat.size()
            content_feat = content_feat[seg]
        else:
            content_feat = content_feat_ori.clone()
            size = content_feat.size()

        feat_var = content_feat.view(1, 4, -1).var(dim=2) + 1e-5
        feat_std = feat_var.sqrt().view(1, 4, 1)
        feat_mean = content_feat.view(1, 4, -1).mean(dim=2).view(1, 4, 1)
        normalized_feat = (content_feat.view(1, 4, -1) - feat_mean) / feat_std
        style_mean, style_std = self.calc_mean_std(style_feat.clone())    
        content_feat = normalized_feat * style_std.squeeze(-1) + style_mean.squeeze(-1)
        content_out = content_feat_ori.clone()
    
        if roi is not None:
            content_out[:, :, roi[0]:roi[1], roi[2]:roi[3]][seg] = alpha * content_feat.flatten() + (1-alpha) * content_feat_ori[:, :, roi[0]:roi[1], roi[2]:roi[3]][seg]
        else:
            content_out = alpha * content_feat + (1-alpha) * content_feat_ori

        return content_out    

    # Adapted from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.encode_prompt
    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        exceptional: bool = False,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            exceptional (bool, defaults to `False`):
                If `True`, the function will perform exceptional prompt inversion.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_input_ids = torch.zeros_like(text_input_ids) + 7788 if exceptional else text_input_ids

            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask, exceptional=exceptional)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True, exceptional=exceptional
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_input.input_ids

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids.to(device),
                attention_mask=attention_mask,
                exceptional=exceptional
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds


    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        composition: bool = False,
        inv_prompt_embeds: Optional[torch.FloatTensor] = None,
        inv_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        inv_negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        inv_negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        inv_guidance_scale: float = 4.5,
        obj_params: Optional[List[Dict]] = None,
        tprime: int = 10,
        tau: int = 3,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            composition (`bool` defaults to `False`):
                If set to `True`, composition process will be conducted intertwine with denoising the background and 
                foreground latents.
            inv_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings for inversion. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            inv_negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings for inversion. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            inv_guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale for inversion
            obj_params (`List[Dict]`, *optional*):
                List of dictionaries containing parameters for compositing objects
            tprime ('int', *optional*):
                The value of T' to initiate composition process
            tau ('int', *optional*):
                The value of tau to constrain number of steps to apply adaptive normalization and energy-guided optimization
        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        with torch.no_grad():
            callback = kwargs.pop("callback", None)
            callback_steps = kwargs.pop("callback_steps", None)

            if callback is not None:
                deprecate(
                    "callback",
                    "1.0.0",
                    "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )
            if callback_steps is not None:
                deprecate(
                    "callback_steps",
                    "1.0.0",
                    "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
                )

            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor
            # to deal with lora scaling and other possible forward hooks

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
                callback_steps,
                negative_prompt,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._guidance_rescale = guidance_rescale
            self._clip_skip = clip_skip
            self._cross_attention_kwargs = cross_attention_kwargs

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # 3. Encode input prompt
            lora_scale = (
                self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
            )

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                exceptional=not composition
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                if composition:
                    inv_prompt_embeds = torch.cat([inv_negative_prompt_embeds, inv_prompt_embeds], dim=0)

            if ip_adapter_image is not None:
                image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, device, num_images_per_prompt)
                if self.do_classifier_free_guidance:
                    image_embeds = torch.cat([negative_image_embeds, image_embeds])

            # 4. Prepare timesteps
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
            if composition:
                latents = latents.chunk(len(obj_params)+1, dim=0)
                bg_latents, all_fg_latents = latents[0], list(latents[1:])   

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 6.1 Add image embeds for IP-Adapter
            added_cond_kwargs = {"image_embeds": image_embeds} if ip_adapter_image is not None else None

            # 6.2 Optionally get Guidance Scale Embedding
            timestep_cond = None
            if self.unet.config.time_cond_proj_dim is not None:
                guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
                timestep_cond = self.get_guidance_scale_embedding(
                    guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
                ).to(device=device, dtype=latents.dtype)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if not composition:
                    with torch.no_grad():
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if self.do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    # Conduct initiation and adaptive normalization
                    blended_latents = bg_latents.clone()
                    alphas = [0.5/tau*(j+1) for j in range(tau)]
                    for fg_latents, params in zip(all_fg_latents, obj_params):
                        rt, rb, rl, rr = params["ref_bbox"]
                        lt, lb, ll, lr = params["latent_bbox"]
                        seg = params["seg"][:, :, rt:rb, rl:rr].bool()
                        if i == tprime or i == 0:
                            blended_latents[:, :, lt:lb, ll:lr][seg] = fg_latents[:, :, rt:rb, rl:rr][seg].clone()
                            composition_latents = blended_latents.clone()
                        elif tprime < i <= tprime + tau:
                            blended_latents[:, :, lt:lb, ll:lr][seg] = composition_latents[:, :, lt:lb, ll:lr][seg].clone()
                            blended_latents = self.adaptive_normalization(blended_latents, bg_latents, alpha=alphas[i-tprime-1], seg=seg, roi=params["latent_bbox"])
                            composition_latents = blended_latents.clone()

                    # Denoising for inverted background and foreground latents
                    with torch.no_grad():
                        bg_latent_model_input = torch.cat([bg_latents] * 2) if self.do_classifier_free_guidance else bg_latents
                        bg_latent_model_input = self.scheduler.scale_model_input(bg_latent_model_input, t)

                        all_fg_latent_model_input = [torch.cat([fg_latents] * 2) if self.do_classifier_free_guidance else fg_latents for fg_latents in all_fg_latents]
                        all_fg_latent_model_input = [self.scheduler.scale_model_input(fg_latent_model_input, t) for fg_latent_model_input in all_fg_latent_model_input]

                        # predict the noise residual
                        bg_noise_pred = self.unet(
                            bg_latent_model_input,
                            t,
                            encoder_hidden_states=inv_prompt_embeds,
                            timestep_cond=timestep_cond,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        all_fg_noise_pred = [
                            self.unet(
                                fg_latent_model_input,
                                t,
                                encoder_hidden_states=inv_prompt_embeds,
                                timestep_cond=timestep_cond,
                                cross_attention_kwargs=self.cross_attention_kwargs,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0] for fg_latent_model_input in all_fg_latent_model_input
                        ]

                        # Perform guidance
                        if self.do_classifier_free_guidance:
                            bg_noise_pred_uncond, bg_noise_pred_text = bg_noise_pred.chunk(2)
                            bg_noise_pred = bg_noise_pred_uncond + inv_guidance_scale * (bg_noise_pred_text - bg_noise_pred_uncond)

                            for j, fg_noise_pred in enumerate(all_fg_noise_pred):
                                fg_noise_pred_uncond, fg_noise_pred_text = fg_noise_pred.chunk(2)
                                all_fg_noise_pred[j] = fg_noise_pred_uncond + inv_guidance_scale * (fg_noise_pred_text - fg_noise_pred_uncond)

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            bg_noise_pred = rescale_noise_cfg(bg_noise_pred, bg_noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # Conduct energy-guided optimization
                    if tprime < i <= tprime + tau:
                        composition_latents = composition_latents.requires_grad_(True)
                        num_opt_step = 3
                    else:
                        composition_latents = composition_latents.detach()
                        num_opt_step = 0
                        
                    for j in range(num_opt_step+1):
                        composition_latent_model_input = torch.cat([composition_latents] * 2) if self.do_classifier_free_guidance else composition_latents
                        composition_latent_model_input = self.scheduler.scale_model_input(composition_latent_model_input, t)

                        composition_noise_pred = self.unet(
                            composition_latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=self.cross_attention_kwargs,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]         
                            
                        if self.do_classifier_free_guidance:
                            composition_noise_pred_uncond, composition_noise_pred_text = composition_noise_pred.chunk(2)
                            composition_noise_pred = composition_noise_pred_uncond + self.guidance_scale * (composition_noise_pred_text - composition_noise_pred_uncond)

                        if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                            composition_noise_pred = rescale_noise_cfg(composition_noise_pred, composition_noise_pred_text, guidance_rescale=self.guidance_rescale)

                        noise_pred = torch.cat([bg_noise_pred] + all_fg_noise_pred + [composition_noise_pred])
                        latents = torch.cat([bg_latents] + all_fg_latents + [composition_latents])

                        prev_x0_latents = [o.clone() if o is not None else None for o in self.scheduler.model_outputs]
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        if j < num_opt_step:
                            x0_latents = self.scheduler.model_outputs[-1].chunk(len(obj_params)+2, dim=0)
                            bg_x0_latents, all_fg_x0_latents, composition_x0_latents = x0_latents[0], list(x0_latents[1:-1]), x0_latents[-1]

                            composition_image = self.vae.decode(
                                composition_x0_latents.to(self.vae.device) / self.vae.config.scaling_factor, return_dict=False)[0]
                                
                            emb_dist, gram_dist = self.clip.get_residual(composition_image, obj_params)
                            loss = emb_dist * 15 + gram_dist * 0.15
                                
                            grad = torch.autograd.grad(outputs=loss, inputs=composition_latents)[0].detach()
                            grad_clone = torch.zeros_like(grad)

                            for params in obj_params:
                                rt, rb, rl, rr = params["ref_bbox"]
                                lt, lb, ll, lr = params["latent_bbox"]
                                seg = params["seg"][:, :, rt:rb, rl:rr].bool()
                                grad_clone[:, :, lt:lb, ll:lr][seg] = grad[:, :, lt:lb, ll:lr][seg]
                        
                            composition_latents = composition_latents - grad_clone
                            self.scheduler.model_outputs = prev_x0_latents
                            self.scheduler._step_index -= 1
                        else:
                            latents = latents.chunk(len(obj_params)+2, dim=0)
                            bg_latents, all_fg_latents, composition_latents = latents[0], list(latents[1:-1]), latents[-1]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.latents_to_img(composition_latents)
            prompt_inputs = None
        else:
            image = latents
            prompt_inputs = (
                prompt_embeds.chunk(2, dim=0)[1] if self.do_classifier_free_guidance else prompt_embeds,
                None,
                negative_prompt_embeds,
                None
            )

        return (image, prompt_inputs)