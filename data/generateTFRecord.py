import argparse

from classificationDataTool import ClassificationImageData


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='from which to generate TFRecord, folders or mxrec', default='mxrec')
    parser.add_argument('--image_size', type=int, help='image size', default=112)
    parser.add_argument('--read_dir', type=str, help='directory to read data', default='')
    parser.add_argument('--save_path', type=str, help='path to save TFRecord file', default='')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cid = ClassificationImageData(img_size=args.image_size)
    if args.mode == 'folders':
        cid.write_tfrecord_from_folders(args.read_dir, args.save_path)
    elif args.mode == 'mxrec':
        cid.write_tfrecord_from_mxrec(args.read_dir, args.save_path)
    else:
        raise('ERROR: wrong mode (only folders and mxrec are supported)')
