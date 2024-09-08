import argparse
from diffusers import DiffusionPipeline
import torch

def generate_image(prompt, width, height, output_file):
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float32,
        use_safetensors=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    image = pipe(prompt=prompt, width=width, height=height).images[0]

    image.save(output_file)
    print(f"Image saved as {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion")
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=512, help="Width of the output image")
    parser.add_argument("--height", type=int, default=512, help="Height of the output image")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output file name")

    args = parser.parse_args()

    generate_image(args.prompt, args.width, args.height, args.output)

if __name__ == "__main__":
    main()
