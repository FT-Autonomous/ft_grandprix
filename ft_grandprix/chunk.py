import json
from PIL import Image
import os
from math import ceil
import numpy as np
import argparse
import sys
import shutil

def chunk(path, output_dir="chunks", chunk_width=20, chunk_height=20, verbose=True, force=False):
    if os.path.exists(output_dir):
        existing_files = os.listdir(output_dir)
        if len(existing_files) != 0:
            if not force:
                print(f"Refusing to overwrite existing directory `{output_dir}`", file=sys.stderr)
                return
            else:
                if verbose: print(f"`{output_dir}` exists but the force option was specified", file=sys.stderr)
                if "metadata.json" not in existing_files:
                    print("Refusing to overwrite directory without a `metadata.json` (we may not have created it)")
                    return
        shutil.rmtree(output_dir)
            
    os.mkdir(output_dir)

    image = Image.open(path)

    horizontal_chunks = ceil(image.width / chunk_width)
    vertical_chunks = ceil(image.height / chunk_height)
    chunks = []
    
    for i in range(horizontal_chunks):
        for j in range(vertical_chunks):
            cropped = image.crop((
                i * chunk_width,
                j * chunk_height,
                min((i+1) * chunk_width, image.width),
                min((j+1) * chunk_height, image.height)
            ))
            np_cropped = np.asarray(cropped)
            chunk_basename = f"{i:03}x{j:03}.png"
            if np_cropped.sum() > 0:
                if verbose: print(f"Going to produce non-empty chunk {chunk_basename}")
                cropped.save(os.path.join(output_dir, chunk_basename))
                chunks.append([i, j])
            else:
                if verbose: print(f"Not going to produce empty chunk {chunk_basename}")
    
    with open("chunks/metadata.json", "w") as metadata_file:
        metadata = {
            "original_width": image.width,
            "original_height": image.height,
            "chunk_width": chunk_width,
            "chunk_height": chunk_height,
            "horizontal_chunks": horizontal_chunks,
            "vertical_chunks": vertical_chunks,
            "chunks": chunks
        }
        json.dump(metadata, metadata_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="the image file split into chunks", dest="input", required=True)
    parser.add_argument("-o", help="the output directory chunks", dest="output", default="chunks")
    parser.add_argument("-W", help="the chunk height to use", dest="chunk_width", default=20)
    parser.add_argument("-H", help="the chunk width height to use", dest="chunk_height", default=20)
    parser.add_argument("-q", help="supress output", dest="quiet", action="store_const", const=True, default=False)
    parser.add_argument("-f", help="overwrite any directory", dest="force", action="store_const", const=True, default=False)
    args = parser.parse_args()
    chunk(
        args.input,
        output_dir=args.output,
        chunk_width=args.chunk_width,
        chunk_height=args.chunk_height,
        verbose=not args.quiet,
        force=args.force
    )
