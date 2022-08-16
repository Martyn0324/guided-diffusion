class TrainSuperRes:

    def __init__(
        self,
        data_dir,
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
        
        model, diffusion = self.sr_create_model_and_diffusion()
        
        dist_util.setup_dist()
        logger.configure()

        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(self.schedule_sampler, diffusion)

        logger.log("creating data loader...")
        data = load_superres_data(
            self.data_dir,
            self.batch_size,
            large_size=self.large_size,
            small_size=self.small_size,
            class_cond=self.class_cond,
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
        
    def sr_create_model(
        self,
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma,
        class_cond,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        resblock_updown,
        use_fp16):

        _ = small_size  # hack to prevent unused variable

        if large_size == 512:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif large_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif large_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported large size: {large_size}")

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(large_size // int(res))

        return SuperResModel(
            image_size=large_size,
            in_channels=3,
            model_channels=num_channels,
            out_channels=(3 if not learn_sigma else 6),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16)

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
        
        print(noise_schedule, steps)

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

    def sr_create_model_and_diffusion(self):

        model = self.sr_create_model(
            self.large_size,
            self.small_size,
            self.num_channels,
            self.num_res_blocks,
            self.learn_sigma,
            self.class_cond,
            self.use_checkpoint,
            self.attention_resolutions,
            self.num_heads,
            self.num_head_channels,
            self.num_heads_upsample,
            self.use_scale_shift_norm,
            self.dropout,
            self.resblock_updown,
            self.use_fp16)
        

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
