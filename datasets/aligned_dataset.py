from torch.utils.data import Dataset
from PIL import Image
from utils import data_util


class AlignedDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        self.data_paths = sorted(data_util.make_dataset(self.opts.input_path))

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        image = Image.open(data_path).convert('RGB')
        w, h = image.size
        w2 = int(w / 2)
        a_part = image.crop((0, 0, w2, h))
        b_part = image.crop((w2, 0, w, h))
        transform = data_util.get_transform()
        a_part = transform(a_part)
        b_part = transform(b_part)

        image_path = data_path
        if self.opts.direction_start_b:
            from_image = b_part
            to_image = a_part
        else:
            from_image = a_part
            to_image = b_part
        return from_image, to_image, image_path

    def __len__(self):
        return len(self.data_paths)
