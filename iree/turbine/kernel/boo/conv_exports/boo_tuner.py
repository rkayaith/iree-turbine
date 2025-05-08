import argparse
import pathlib
import shlex
from pathlib import Path
import os
import tempfile
import subprocess
import shutil
import traceback

from model_tuner.model_tuner import main as tuner_main
from iree.turbine.kernel.boo.conv_exports import (
    miopen_parser as mio,
    launch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commands-file", type=str, help="read commands from file")
    parser.add_argument("--output-td-spec", type=Path, default="tuning-spec.mlir")
    parser.add_argument("--starter-td-spec", type=Path)
    parser.add_argument("--num-candidates", type=int, default=100)
    args, extra_cli_args = parser.parse_known_args()

    if args.commands_file:
        with open(args.commands_file) as f:
            mio_file_args = [
                shlex.split(s) for s in f.readlines() if not s.startswith("#")
            ]
    else:
        mio_file_args = [[]]  # use CLI arguments

    if args.starter_td_spec not in (None, args.output_td_spec):
        shutil.copyfile(args.starter_td_spec, args.output_td_spec)
    else:
        args.output_td_spec.touch(exist_ok=False)
    for idx, file_args in enumerate(mio_file_args):
        cli_args = file_args + extra_cli_args
        print(f">>> ({idx+1}/{len(mio_file_args)}) {shlex.join(cli_args)}")
        parser = mio.get_miopen_parser()
        parser.add_argument("-t")
        sig = mio.get_signature(parser.parse_args(cli_args))
        asm = launch._get_module_asm(sig)

        tmp_dir = Path(tempfile.mkdtemp(dir=".", prefix="boo-tuner-"))
        input_ir_path = tmp_dir / "boo-model.mlir"
        bench_dir = tmp_dir / "bench"
        with open(input_ir_path, "w") as f:
            f.write(asm)
        subprocess.run(
            [
                "iree-compile",
                input_ir_path,
                "--iree-hal-target-device=hip",
                "--iree-hip-target=gfx942",
                "--iree-config-add-tuner-attributes",
                "--iree-hal-dump-executable-benchmarks-to",
                bench_dir,
                "-o",
                "/dev/null",
            ]
        )
        [bench_file] = os.listdir(bench_dir)
        bench_path = bench_dir / bench_file

        tuner_args = [
            "model-unused",
            str(bench_path),
            *("--starter-td-spec", str(args.output_td_spec)),
            *("--output-td-spec", str(args.output_td_spec)),
            "--compile-flags-file=/home/rkayaith/repos/shark-ai/sharktuner/model_tuner/compile_flags.txt",
            "--devices=hip://0",
            "--model-tuner-num-dispatch-candidates=100",
            f"--num-candidates={args.num_candidates}",
            "--codegen-pipeline=llvmgpu_tile_and_fuse",
            "--stop-after=benchmark-dispatches",
        ]
        print(f"> {shlex.join(tuner_args)}")
        try:
            tuner_main(tuner_args)
            shutil.rmtree(tmp_dir)
        except Exception as err:
            traceback.print_exception(err)


main()
