from magvit2_pytorch import (
    VideoTokenizer
)

from trainer import (
    VideoTokenizerTrainer
)

tokenizer = VideoTokenizer(
    image_size = 384,
    init_dim = 64,
    channels = 1,
    max_dim = 512,
    codebook_size = 1024,
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
        'attend_space',
        'compress_space',
        ('consecutive_residual', 2),
    )
)

trainer = VideoTokenizerTrainer(
    tokenizer,
    dataset_folder = '/path/to/a/lot/of/media',     # folder of either videos or images, depending on setting below
    dataset_type = 'videos',                        # 'videos' or 'images', prior papers have shown pretraining on images to be effective for video synthesis
    batch_size = 4,
    grad_accum_every = 8,
    learning_rate = 2e-5,
    num_train_steps = 1_000_000,
    use_wandb_tracking = True
)

with trainer.trackers(project_name = 'magvit2', run_name = 'baseline'):
    trainer.train()