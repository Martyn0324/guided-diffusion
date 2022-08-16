class TrainNormal:
    
    def __init__(
        self,
        data_dir,
        numpy_data=False,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
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
        small_size=64
    ):
        
        self.data_dir = data_dir
        self.numpy_data = numpy_data
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
        
        model, diffusion = self.create_model_and_diffusion()
        
        dist_util.setup_dist()
        logger.configure()

        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(self.schedule_sampler, diffusion)

        logger.log("creating data loader...")
        data = load_data(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            image_size=self.image_size,
            class_cond=self.class_cond,
            numpy=self.numpy_data,
        )

        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data,
            batch_size=self.batch_size,
            microbatch=self.microbatch,
            lr=self.learning_rate,
            ema_rate=self.ema_rate,
            log_interval=self.log_interval,
            save_interval=self.save_interval,
            resume_checkpoint=self.resume_checkpoint,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=self.weight_decay,
            lr_anneal_steps=self.steps,
        ).run_loop()
        
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
