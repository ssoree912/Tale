from typing import Callable, List, Optional, Tuple, Union, Dict
from torchvision import transforms
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import *
from modeling_t5_exceptional import *

class TALEPixArtAlphaPipeline(PixArtAlphaPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__(
            tokenizer,
            text_encoder,
            vae,
            transformer,
            scheduler,
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
        use_resolution_binning: bool,
        target_size: Union[int, Tuple[int, int]]
    ) -> torch.FloatTensor:
        x = x.to(self.vae.device)
        image = self.vae.decode(x / self.vae.config.scaling_factor, return_dict=False)[0]
        if isinstance(target_size, int):
            w = h = target_size
        else:
            w, h = target_size
        if use_resolution_binning:
            image = self.resize_and_crop_tensor(image, w, h)
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
            content_feat = content_feat_ori[:, :, roi[0]:roi[1], roi[2]:roi[3]][seg].clone()
            size = content_feat.size()
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

        print(seg)

        return content_out

    # Adapted from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        exceptional: bool = False,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            exceptional (bool, defaults to `False`):
                If `True`, the function will perform exceptional prompt inversion. 
        """

        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = 120

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask, exceptional=exceptional)
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask, exceptional=exceptional
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask


    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        use_resolution_binning: bool = True,
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
    ) -> Union[ImagePipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 4.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            use_resolution_binning (`bool` defaults to `True`):
                If set to `True`, the requested height and width are first mapped to the closest resolutions using
                `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
                the requested resolution. Useful for generating non-square images.
            composition (`bool` defaults to `False`):
                If set to `True`, composition process will be conducted intertwine with denoising the background and 
                foreground latents.
            inv_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings for inversion. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            inv_prompt_attention_mask (`torch.FloatTensor`, *optional*): Pre-generated attention mask for text embeddings for inversion.
            inv_negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings for inversion. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            inv_negative_prompt_attention_mask (`torch.FloatTensor`, *optional*):
                Pre-generated attention mask for negative text embeddings for inversion.
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
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        with torch.no_grad():
            if "mask_feature" in kwargs:
                deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
                deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
            # 1. Check inputs. Raise error if not correct
            height = height or self.transformer.config.sample_size * self.vae_scale_factor
            width = width or self.transformer.config.sample_size * self.vae_scale_factor
            if use_resolution_binning:
                aspect_ratio_bin = (
                    ASPECT_RATIO_1024_BIN if self.transformer.config.sample_size == 128 else ASPECT_RATIO_512_BIN
                )
                orig_height, orig_width = height, width
                height, width = self.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

            self.check_inputs(
                prompt,
                height,
                width,
                negative_prompt,
                callback_steps,
                prompt_embeds,
                negative_prompt_embeds,
                prompt_attention_mask,
                negative_prompt_attention_mask,
            )

            # 2. Default height and width to transformer
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            device = self._execution_device

            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask
            ) = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                exceptional=not composition
            )

            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
                if composition:
                    inv_prompt_embeds = torch.cat([inv_negative_prompt_embeds, inv_prompt_embeds], dim=0)
                    inv_prompt_attention_mask = torch.cat([inv_negative_prompt_attention_mask, inv_prompt_attention_mask], dim=0)                    

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latents.
            latent_channels = self.transformer.config.in_channels
        
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                latent_channels,
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

            # 6.1 Prepare micro-conditions.
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if self.transformer.config.sample_size == 128:
                resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
                aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size * num_images_per_prompt, 1)
                resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
                aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)
                added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7. Composition (denoising) loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if not composition:
                    with torch.no_grad():
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        current_timestep = t
                        if not torch.is_tensor(current_timestep):
                            is_mps = latent_model_input.device.type == "mps"
                            if isinstance(current_timestep, float):
                                dtype = torch.float32 if is_mps else torch.float64
                            else:
                                dtype = torch.int32 if is_mps else torch.int64
                            current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                        elif len(current_timestep.shape) == 0:
                            current_timestep = current_timestep[None].to(latent_model_input.device)
                        # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        current_timestep = current_timestep.expand(latent_model_input.shape[0])
                    
                        noise_pred = self.transformer(
                            latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=current_timestep,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        if self.transformer.config.out_channels // 2 == latent_channels:
                            noise_pred = noise_pred.chunk(2, dim=1)[0]

                        # Compute previous image: x_t -> x_t-1
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
                        bg_latent_model_input = torch.cat([bg_latents] * 2) if do_classifier_free_guidance else bg_latents
                        bg_latent_model_input = self.scheduler.scale_model_input(bg_latent_model_input, t)

                        all_fg_latent_model_input = [torch.cat([fg_latents] * 2) if do_classifier_free_guidance else fg_latents for fg_latents in all_fg_latents]
                        all_fg_latent_model_input = [self.scheduler.scale_model_input(fg_latent_model_input, t) for fg_latent_model_input in all_fg_latent_model_input]

                        current_timestep = t
                        if not torch.is_tensor(current_timestep):
                            is_mps = bg_latent_model_input.device.type == "mps"
                            if isinstance(current_timestep, float):
                                dtype = torch.float32 if is_mps else torch.float64
                            else:
                                dtype = torch.int32 if is_mps else torch.int64
                            current_timestep = torch.tensor([current_timestep], dtype=dtype, device=bg_latent_model_input.device)
                        elif len(current_timestep.shape) == 0:
                            current_timestep = current_timestep[None].to(bg_latent_model_input.device)
                        # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        current_timestep = current_timestep.expand(bg_latent_model_input.shape[0])

                        bg_noise_pred = self.transformer(
                            bg_latent_model_input,
                            encoder_hidden_states=inv_prompt_embeds,
                            encoder_attention_mask=inv_prompt_attention_mask,
                            timestep=current_timestep,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        all_fg_noise_pred = [
                            self.transformer(
                                fg_latent_model_input,
                                encoder_hidden_states=inv_prompt_embeds,
                                encoder_attention_mask=inv_prompt_attention_mask,
                                timestep=current_timestep,
                                added_cond_kwargs=added_cond_kwargs,
                                return_dict=False,
                            )[0] for fg_latent_model_input in all_fg_latent_model_input
                        ]

                        # Perform guidance
                        if do_classifier_free_guidance:
                            bg_noise_pred_uncond, bg_noise_pred_text = bg_noise_pred.chunk(2)
                            bg_noise_pred = bg_noise_pred_uncond + inv_guidance_scale * (bg_noise_pred_text - bg_noise_pred_uncond)

                            for j, fg_noise_pred in enumerate(all_fg_noise_pred):
                                fg_noise_pred_uncond, fg_noise_pred_text = fg_noise_pred.chunk(2)
                                all_fg_noise_pred[j] = fg_noise_pred_uncond + inv_guidance_scale * (fg_noise_pred_text - fg_noise_pred_uncond)

                        if self.transformer.config.out_channels // 2 == latent_channels:
                            bg_noise_pred = bg_noise_pred.chunk(2, dim=1)[0]
                            for j, fg_noise_pred in enumerate(all_fg_noise_pred):
                                all_fg_noise_pred[j] = fg_noise_pred.chunk(2, dim=1)[0]

                    # Conduct energy-guided optimization
                    if tprime < i <= tprime + tau:
                        composition_latents = composition_latents.requires_grad_(True)
                        num_opt_step = 3
                    else:
                        composition_latents = composition_latents.detach()
                        num_opt_step = 0

                    for j in range(num_opt_step+1):
                        composition_latent_model_input = torch.cat([composition_latents] * 2) if do_classifier_free_guidance else composition_latents
                        composition_latent_model_input = self.scheduler.scale_model_input(composition_latent_model_input, t)

                        composition_noise_pred = self.transformer(
                            composition_latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            encoder_attention_mask=prompt_attention_mask,
                            timestep=current_timestep,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]           

                        if do_classifier_free_guidance:
                            composition_noise_pred_uncond, composition_noise_pred_text = composition_noise_pred.chunk(2)
                            composition_noise_pred = composition_noise_pred_uncond + guidance_scale * (composition_noise_pred_text - composition_noise_pred_uncond)                                            


                        if self.transformer.config.out_channels // 2 == latent_channels:
                            composition_noise_pred = composition_noise_pred.chunk(2, dim=1)[0]

                        noise_pred = torch.cat([bg_noise_pred] + all_fg_noise_pred + [composition_noise_pred])
                        latents = torch.cat([bg_latents] + all_fg_latents + [composition_latents])

                        # Compute previous image: x_t -> x_t-1
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

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.latents_to_img(composition_latents, use_resolution_binning, (orig_width, orig_height))
            prompt_inputs = None
        else:
            image = latents
            prompt_inputs = (
                prompt_embeds.chunk(2, dim=0)[1] if do_classifier_free_guidance else prompt_embeds,
                prompt_attention_mask.chunk(2, dim=0)[1] if do_classifier_free_guidance else prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask
            )

        # Offload all models
        self.maybe_free_model_hooks()

        return (image, prompt_inputs)