#!/usr/bin/env bash
.venv/bin/python data_processor/nio_processor/nio_process.py \
    --raw_tar /home/nio/Projects/gs_rec/data/nop/efa3fa0a-4af8-4f67-846e-24116d7470d6/processed-clip-data.tar \
    --save_dir data/nio/nop/efa3fa0a-4af8-4f67-846e-24116d7470d6 \
    --depth_decoder /home/nio/.cache/drivegs/depth_decoder/5505691d8d97178a/dump_depth_elm \
    --workers 4 --render_workers 2