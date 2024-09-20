from magvit2_pytorch import (
    VideoTokenizer
)

from trainer import (
    VideoTokenizerTrainer
)

tokenizer = VideoTokenizer(
    image_size = 384,
    init_dim = 128,
    channels = 1,
    max_dim = 512,
    # codebook_size = 4096,
    fsq_levels = [8, 5, 5, 5, 4],
    use_fsq = True,
    use_gan = True,
    perceptual_loss_weight = 0,
    layers = (
        'residual',
        'compress_space',
        ('consecutive_residual', 2),
        'linear_attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'linear_attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'attend_space',
        'compress_space',
        ('consecutive_residual', 2),
        'compress_space',
        ('consecutive_residual', 2),
        'attend_space'
    )
)

trainer = VideoTokenizerTrainer(
    tokenizer,
    dataset_folder = '/path/to/a/lot/of/media',     # folder of either videos or images, depending on setting below
    dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 2,
    grad_accum_every = 2,
    accelerate_kwargs={"split_batches": True},
    learning_rate = 0.08e-3,
    num_train_steps = 1_000_000,
    discr_start_after_step = 0,
    use_wandb_tracking = True
)

with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
    trainer.train()