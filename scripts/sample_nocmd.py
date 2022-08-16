class TestNormal:
    
    def __init__(
        self,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        large_size=256,
        small_size=64,
        clip_denoised=True,
        num_samples=16,
        use_ddim=False,
        base_samples="",
        model_path="",
    ):
        
        self.model_path = model_path
        self.schedule_sampler=schedule_sampler
        self.learning_rate = lr
        self.weight_decay=weight_decay
        self.steps = lr_anneal_steps
        self.batch_size=batch_size
        self.microbatch=microbatch
        self.ema_rate=ema_rate
        self.log_interval=log_interval
        self.save_interval=save_interval
        self.resume_checkpoint=resume_checkpoint
        self.use_fp16=use_fp16
        self.fp16_scale_growth=fp16_scale_growth
        self.image_size=image_size
        self.num_channels=num_channels
        self.num_res_blocks=num_res_blocks
        self.num_heads=num_heads
        self.num_heads_upsample=num_heads_upsample
        self.num_head_channels=num_head_channels
        self.attention_resolutions=attention_resolutions
        self.channel_mult=channel_mult
        self.dropout=dropout
        self.class_cond=class_cond
        self.use_checkpoint=use_checkpoint
        self.use_scale_shift_norm=use_scale_shift_norm
        self.resblock_updown=resblock_updown
        self.use_new_attention_order=use_new_attention_order
        self.learn_sigma=learn_sigma
        self.diffusion_steps=diffusion_steps
        self.noise_schedule=noise_schedule
        self.timestep_respacing=timestep_respacing
        self.use_kl=use_kl
        self.predict_xstart=predict_xstart
        self.rescale_timesteps=rescale_timesteps
        self.rescale_learned_sigmas=rescale_learned_sigmas
        self.large_size=large_size
        self.small_size=small_size
        self.clip_denoised=clip_denoised
        self.num_samples=num_samples
        self.use_ddim=use_ddim
        self.base_samples=base_samples
        
        
        dist_util.setup_dist()
        logger.configure()

        logger.log("creating model and diffusion...")
        
        model, diffusion = self.create_model_and_diffusion()
        
        model.load_state_dict(
            dist_util.load_state_dict(self.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if self.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("sampling...")
        
        all_images = []
        all_labels = []
        while len(all_images) * self.batch_size < self.num_samples:
            model_kwargs = {}
            if self.class_cond:
                classes = th.randint(
                    low=0, high=NUM_CLASSES, size=(self.batch_size,), device=dist_util.dev()
                )
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not self.use_ddim else diffusion.ddim_sample_loop
            )
            sample = sample_fn(
                model,
                (self.batch_size, 3, self.image_size, self.image_size),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if self.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * self.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: self.num_samples]
        if self.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: self.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            #out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            out_path = os.path.join(f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if self.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")
        
        

    def create_model(
        self,
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16",
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False):

        if channel_mult == "":
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        return UNetModel(
            image_size=image_size,
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order)

    def create_gaussian_diffusion(
        self,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing=""):


        betas = get_named_beta_schedule(noise_schedule, steps)
        if use_kl:
            loss_type = LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = LossType.RESCALED_MSE
        else:
            loss_type = LossType.MSE
        if not timestep_respacing:
            timestep_respacing = [steps]
        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )

    def create_model_and_diffusion(self):

        model = self.create_model(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_res_blocks=self.num_res_blocks,
            channel_mult=self.channel_mult,
            learn_sigma=self.learn_sigma,
            class_cond=self.class_cond,
            use_checkpoint=self.use_checkpoint,
            attention_resolutions=self.attention_resolutions,
            num_heads=self.num_heads,
            num_head_channels=self.num_head_channels,
            num_heads_upsample=self.num_heads_upsample,
            use_scale_shift_norm=self.use_scale_shift_norm,
            dropout=self.dropout,
            resblock_updown=self.resblock_updown,
            use_fp16=self.use_fp16,
            use_new_attention_order=self.use_new_attention_order)

        diffusion = self.create_gaussian_diffusion(
            steps=self.diffusion_steps,
            learn_sigma=self.learn_sigma,
            noise_schedule=self.noise_schedule,
            use_kl=self.use_kl,
            predict_xstart=self.predict_xstart,
            rescale_timesteps=self.rescale_timesteps,
            rescale_learned_sigmas=self.rescale_learned_sigmas,
            timestep_respacing=self.timestep_respacing)

        return model, diffusion
