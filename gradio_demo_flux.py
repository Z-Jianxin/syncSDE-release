import torch
import gradio as gr
from PIL import ImageChops, Image

from pipeline_syncsde import RFInversionFluxPipelineSDE
from pipeline_syncsde_flux2 import Flux2SyncSDEPipeline


FLUX1_KEY = "flux1"
FLUX2_KEY = "flux2"
MODEL_LABELS = {
    FLUX1_KEY: "FLUX.1-dev (syncSDE)",
    FLUX2_KEY: "FLUX.2-dev (syncSDE)",
}
MODEL_KEYS_BY_LABEL = {label: key for key, label in MODEL_LABELS.items()}
FLUX1_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX2_MODEL_ID = "black-forest-labs/FLUX.2-dev"
FLUX2_QUANTIZED_MODEL_ID = "diffusers/FLUX.2-dev-bnb-4bit"


def normalize_model_key(model):
    return MODEL_KEYS_BY_LABEL.get(model, model)


class FluxEditor:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        flux1_model_id=FLUX1_MODEL_ID,
        flux2_model_id=FLUX2_MODEL_ID,
        flux2_quantized_model_id=FLUX2_QUANTIZED_MODEL_ID,
        flux2_4bit=True,
    ):
        self.device = device
        self.flux1_model_id = flux1_model_id
        self.flux2_model_id = flux2_model_id
        self.flux2_quantized_model_id = flux2_quantized_model_id
        self.flux2_4bit = flux2_4bit
        self.pipes = {}
        self.load_pipe(FLUX1_KEY)

    def load_pipe(self, model_key):
        model_key = normalize_model_key(model_key)
        if model_key in self.pipes:
            return self.pipes[model_key]

        if model_key == FLUX1_KEY:
            pipe = RFInversionFluxPipelineSDE.from_pretrained(
                self.flux1_model_id,
                torch_dtype=torch.bfloat16,
            )
        elif model_key == FLUX2_KEY:
            if self.flux2_4bit:
                from diffusers import Flux2Transformer2DModel
                from transformers import Mistral3ForConditionalGeneration

                transformer = Flux2Transformer2DModel.from_pretrained(
                    self.flux2_quantized_model_id,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                )
                text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                    self.flux2_quantized_model_id,
                    subfolder="text_encoder",
                    dtype=torch.bfloat16,
                    device_map=self.device,
                )
                pipe = Flux2SyncSDEPipeline.from_pretrained(
                    self.flux2_quantized_model_id,
                    transformer=transformer,
                    text_encoder=text_encoder,
                    torch_dtype=torch.bfloat16,
                )
            else:
                pipe = Flux2SyncSDEPipeline.from_pretrained(
                    self.flux2_model_id,
                    torch_dtype=torch.bfloat16,
                )
        else:
            raise ValueError(f"Unknown model: {model_key}")

        if model_key == FLUX2_KEY and self.flux2_4bit and self.device.startswith("cuda"):
            pipe.vae.to(self.device)

        self.pipes[model_key] = pipe
        return pipe

    def edit(
        self,
        model,
        init_image,
        source_prompt,
        source_guidance,
        target_prompt,
        target_guidance,
        num_steps,
        starting_index,
        resample,
        max_embed_length,

        source_image_prompt_2,
        target_prompt_2,
        negative_prompt,
        negative_prompt_2,
        seed,
    ):
        model_key = normalize_model_key(model)
        pipe = self.load_pipe(model_key)
        flux2_resident = model_key == FLUX2_KEY and self.flux2_4bit
        if not flux2_resident:
            pipe = pipe.to(self.device)
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if not seed:
            seed = torch.Generator(device="cpu").seed()
        print("Random seed: ", seed)
        torch.manual_seed(int(seed))
        
        source_image_prompt_2 = source_image_prompt_2 if source_image_prompt_2 else None
        target_prompt_2 = target_prompt_2 if target_prompt_2 else None
        negative_prompt = negative_prompt if negative_prompt else None
        negative_prompt_2 = negative_prompt_2 if negative_prompt_2 else None

        pipe_kwargs = dict(
            image=init_image,
            starting_index=int(starting_index), # 0 or 1 can be very numerically unstable
            prompt=target_prompt,
            source_image_prompt=source_prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=target_guidance,
            source_guidance_scale=source_guidance,
            strength=1.0,
            resample=resample,
            max_sequence_length=int(max_embed_length),
        )
        if model_key == FLUX1_KEY:
            pipe_kwargs.update(
                mask=None,
                prompt_2=target_prompt_2,
                source_image_prompt_2=source_image_prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
            )
        edited_image = pipe(**pipe_kwargs).images[0]
        init_resized = init_image.resize(
            edited_image.size,              # match W×H
            Image.Resampling.LANCZOS        # high-quality down/up-sampling
        )
        diff_img = ImageChops.difference(
            init_resized.convert("RGB"),   # make sure both are RGB
            edited_image.convert("RGB")
        )
        if not flux2_resident:
            self.pipes[model_key] = pipe.to("cpu")

        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        print("End Edit\n\n")
        return edited_image, diff_img



def create_demo(
    model_name: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    flux2_4bit: bool = True,
):
    editor = FluxEditor(device=device, flux2_4bit=flux2_4bit)

    with gr.Blocks() as demo:
        gr.Markdown(model_name)
        
        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    label="Model",
                    choices=list(MODEL_LABELS.values()),
                    value=MODEL_LABELS[FLUX1_KEY],
                )
                source_prompt = gr.Textbox(label="Source Prompt", value="")
                target_prompt = gr.Textbox(label="Target Prompt", value="")

                generate_btn = gr.Button("Generate")
                
                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 100, 28, step=1, label="Number of steps")
                    source_guidance = gr.Slider(0.0, 10.0, 1.0, step=0.05, label="Source Guidance: CFG strength for source image")
                    target_guidance = gr.Slider(0.0, 10.0, 1.0, step=0.05, label="Target Guidance: CFG strength for target image")
                    starting_index = gr.Slider(0, 100, 4, step=1, label="starting index, smaller for better structural edit")
                    resample = gr.Checkbox(label="resample the reference process?", value=False,)
                    max_embed_length = gr.Textbox(512, label="max embedding length to hold texts",)
                    seed = gr.Textbox(None, label="Seed")
                    source_image_prompt_2 = gr.Textbox(label="Source Prompt 2 for t5 encoder", value=None)
                    target_prompt_2 = gr.Textbox(label="Target Prompt 2 for t5 encoder", value=None)
                    negative_prompt = gr.Textbox(label="Negative Prompt", value=None)
                    negative_prompt_2 = gr.Textbox(label="Negative Prompt 2", value=None)
            
            with gr.Column():
                init_image = gr.Image(label="Input Image", visible=True, type='pil')
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image", format='jpg')
            
            with gr.Column():
                diff_image   = gr.Image(label="Difference (|input - output|)", format='jpg')

        generate_btn.click(
            fn=editor.edit,
            inputs=[        
                model,
                init_image,
                source_prompt,
                source_guidance,
                target_prompt,
                target_guidance,
                num_steps,
                starting_index,
                resample,
                max_embed_length,
                
                source_image_prompt_2,
                target_prompt_2,
                negative_prompt,
                negative_prompt_2,

                seed,
            ],
            outputs=[output_image, diff_image]
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--no-flux2-4bit", action="store_true", help="Load FLUX.2 in BF16 instead of 4-bit")

    parser.add_argument("--port", type=int, default=41035)
    args = parser.parse_args()

    demo = create_demo("SDE coupling Demo with Flux", args.device, not args.no_flux2_4bit)
    demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
