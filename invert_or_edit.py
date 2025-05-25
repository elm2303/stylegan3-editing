import sys
import argparse
import numpy as np
import torch
import pyrallis
from torch.utils.data import DataLoader
from pathlib import Path
from detect_angle import get_closest_yaw, prepare_factor_ranges_dynamic

from inversion.options.test_options import TestOptions
from utils.inference_utils import load_encoder, get_average_image, run_on_batch
from utils.common import tensor2im
from inversion.datasets.inference_dataset import InferenceDataset
from editing.interfacegan.face_editor import FaceEditor
from models.stylegan3.model import GeneratorType

def invert_all(net, opts, test_opts):
    dataset = InferenceDataset(
        root=opts.data_path,
        transform=opts.data_transforms['transform_inference']
    )
    loader = DataLoader(
        dataset,
        batch_size=opts.test_batch_size,
        shuffle=False,
        num_workers=int(opts.test_workers),
        drop_last=False
    )

    collected_latents = []
    image_names = []
    out_inv_dir = Path(test_opts.output_path) / 'inversions'
    out_inv_dir.mkdir(exist_ok=True, parents=True)

    for idx, (inputs, _) in enumerate(loader):
        with torch.no_grad():
            result_batch, result_latents = run_on_batch(
                inputs=inputs.cuda().float(),
                net=net,
                opts=opts,
                avg_image=get_average_image(net)
            )
            for b in range(len(result_batch)):
                img_path = dataset.paths[idx * opts.test_batch_size + b]
                base = img_path.stem
                ext = img_path.suffix or '.png'
                inv_tensor = result_batch[b][-1]
                inv_img = tensor2im(inv_tensor)
                inv_img.save(out_inv_dir / f"{base}_inversion{ext}")
                collected_latents.append(torch.from_numpy(result_latents[b][-1]))
                image_names.append(str(img_path.name))

    all_latents = torch.stack(collected_latents, dim=0).cpu().numpy()
    # Save latents and original filenames into NPZ
    np.savez(
        test_opts.output_path / 'inversion_data.npz',
        latents=all_latents,
        image_names=np.array(image_names)
    )
    print(f"Saved {all_latents.shape[0]} latents and inversion images to '{out_inv_dir}'")
    
def edit_all(net, test_opts, factor_ranges):
    data = np.load(Path(test_opts.output_path) / 'inversion_data.npz')
    latents = torch.from_numpy(data['latents']).cuda()
    num_imgs = latents.shape[0]
    
    if len(factor_ranges) != num_imgs:
        raise ValueError(
            f"Need exactly one factor_range per image: "
            f"found {len(factor_ranges)} ranges for {num_imgs} images."
        )

    image_names = (
        list(data['image_names'])
        if 'image_names' in data.files
        else [f"img_{i:05d}.png" for i in range(num_imgs)]
    )

    editor = FaceEditor(net.decoder, generator_type=GeneratorType.ALIGNED)
    base_out = Path(test_opts.output_path) / 'editing_results'
    base_out.mkdir(parents=True, exist_ok=True)

    for idx, (img_name, frange) in enumerate(zip(image_names, factor_ranges)):
        latent = latents[idx:idx+1]   # single-image batch
        stem   = Path(img_name).stem
        out_dir = base_out / stem
        out_dir.mkdir(exist_ok=True)

        edited_images, _ = editor.edit(
            latents=latent,
            direction="pose",
            factor_range=frange,
            apply_user_transformations=False
        )

        for step_idx, step in enumerate(edited_images):
            img = step[0]  # get the single PIL.Image out of the list
            img.save(out_dir / f"{stem}_step{step_idx}.png")

    print("All edits completed.")

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=['invert', 'edit'], required=True,
                        help="Choose 'invert' to run inversion, 'edit' to run editing")
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    @pyrallis.wrap()
    def run(test_opts: TestOptions):
        # Load encoder and generator once
        net, opts = load_encoder(checkpoint_path=test_opts.checkpoint_path,
                                 test_opts=test_opts)
        dataset_args = opts.dataset_type and None
        try:
            dataset_args = __import__('configs.data_configs', fromlist=['DATASETS']).DATASETS[opts.dataset_type]
            opts.data_transforms = dataset_args['transforms'](opts).get_transforms()
        except Exception:
            opts.data_transforms = {}

        if args.mode == 'invert':
            invert_all(net, opts, test_opts)
        elif args.mode == 'edit':
            factor_ranges = prepare_factor_ranges_dynamic()
            edit_all(net, test_opts, factor_ranges)
            editing_results = Path(test_opts.output_path) / 'editing_results'
            get_closest_yaw(editing_results)
        else:
            raise ValueError(f"Unknown mode {args.mode}")
    run()


if __name__ == '__main__':
    main()
