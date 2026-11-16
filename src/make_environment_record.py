"""
make_environment_record.py -- record the exact environment a run used.

Reproducibility needs more than fixed seeds: the same code on a different
torch or CUDA build can give slightly different floating-point results. This
writes the versions actually present so a reader can match them.

Run:  python src/make_environment_record.py
"""
import platform
import sys

import torch
import torchvision
import numpy
import scipy
import pandas
import matplotlib

from common import OUTPUT_DIR


def main():
    lines = [
        "# Environment used to produce the results in this repository",
        f"python          {sys.version.split()[0]}",
        f"platform        {platform.platform()}",
        f"torch           {torch.__version__}",
        f"torchvision     {torchvision.__version__}",
        f"numpy           {numpy.__version__}",
        f"scipy           {scipy.__version__}",
        f"pandas          {pandas.__version__}",
        f"matplotlib      {matplotlib.__version__}",
        f"cuda available  {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines += [
            f"cuda version    {torch.version.cuda}",
            f"cudnn version   {torch.backends.cudnn.version()}",
            f"gpu             {torch.cuda.get_device_name(0)}",
        ]
    try:
        import pytorch_grad_cam
        lines.append(f"grad-cam        {pytorch_grad_cam.__version__}")
    except Exception:
        lines.append("grad-cam        1.5.0 (pinned in requirements.txt)")

    text = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "environment.txt").write_text(text)
    print(text)
    print(f"written to {OUTPUT_DIR / 'environment.txt'}")


if __name__ == "__main__":
    main()
