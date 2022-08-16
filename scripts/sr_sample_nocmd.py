class TestSuperRes:
    
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
        num_samples=10000,
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
        
        
        model, diffusion = sr_create_model_and_diffusion()

        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(self.schedule_sampler, diffusion)
        
        model.load_state_dict(
            dist_util.load_state_dict(self.model_path, map_location="cpu")
        )
        model.to(dist_util.dev())
        if self.use_fp16:
            model.convert_to_fp16()
        model.eval()

        logger.log("loading data...")
        data = load_data_for_worker(self.base_samples, self.batch_size, self.class_cond)

        logger.log("creating samples...")
        all_images = []
        while len(all_images) * self.batch_size < self.num_samples:
            model_kwargs = next(data)
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
            sample = diffusion.p_sample_loop(
                model,
                (self.batch_size, 3, self.large_size, self.large_size),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            all_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(all_samples, sample)  # gather not supported with NCCL
            for sample in all_samples:
                all_images.append(sample.cpu().numpy())
            logger.log(f"created {len(all_images) * self.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr)

        dist.barrier()
        logger.log("sampling complete")
        


    def load_data_for_worker(base_samples, batch_size, class_cond):
        with bf.BlobFile(base_samples, "rb") as f:
            obj = np.load(f)
            image_arr = obj["arr_0"]
            if class_cond:
                label_arr = obj["arr_1"]
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        buffer = []
        label_buffer = []
        while True:
            for i in range(rank, len(image_arr), num_ranks):
                buffer.append(image_arr[i])
                if class_cond:
                    label_buffer.append(label_arr[i])
                if len(buffer) == batch_size:
                    batch = th.from_numpy(np.stack(buffer)).float()
                    batch = batch / 127.5 - 1.0
                    batch = batch.permute(0, 3, 1, 2)
                    res = dict(low_res=batch)
                    if class_cond:
                        res["y"] = th.from_numpy(np.stack(label_buffer))
                    yield res
                    buffer, label_buffer = [], []
