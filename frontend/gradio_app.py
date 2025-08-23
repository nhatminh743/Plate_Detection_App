
#-------------------------------------------- IMPORT LIBRARIES ---------------------------------------------------------
import gradio as gr
import requests
from PIL import Image
import io
import base64
import traceback
#-------------------------------------------------  EXTERNAL LINKS  ----------------------------------------------------
# css_fileLink = r'/home/minhpn/Desktop/Green_Parking/gradio_path/styles.css'
#--------------------------------------------------- FUNCTIONS ---------------------------------------------------------
def upload_and_return_prediction_filepath(file_paths):

    url = "http://127.0.0.1:8000/upload-files-multiple"

    files = [
        ("files", (file_path.split("/")[-1], open(file_path, "rb"), "image/jpeg"))
        for file_path in file_paths
    ]

    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        data = response.json()  # This is a list

        text_data = data[0]  # The first item is the text dictionary

        result_lines = []
        for filename, info in text_data.items():
            text_list = info.get("text", [])
            if isinstance(text_list, list):
                text = ", ".join(text_list)  # ✅ Removes brackets and quotes
            else:
                text = str(text_list)
            result_lines.append(f"{filename}: {text}")

        return "\n".join(result_lines)

    except Exception as e:
        return f"❌ Error: {str(e)}"


def upload_and_return_prediction_image(file_path):
    url = "http://127.0.0.1:8000/upload-files-single"

    try:
        with open(file_path, 'rb') as f:
            files = {
                "file": (file_path.split("/")[-1], f, "image/jpeg")
            }

            response = requests.post(url, files=files)
            response.raise_for_status()
        data = response.json()


        results = data.get("text", "No text detected.")
        dimension = data.get("dimension_of_plate", [])
        image_base64 = data.get("image", "")

        if not image_base64:
            return results, None

        #
        image_data = image_base64.split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        text_data = data.get("text", {})

        result_lines = []
        for filename, info in text_data.items():
            text_list = info.get("text", [])
            if isinstance(text_list, list):
                text = ", ".join(text_list)
            else:
                text = str(text_list)
            result_lines.append(f"{text}")

        return "\n".join(result_lines), image

    except Exception as e:
        print(traceback.format_exc())
        return f"❌ Error: {str(e)}", None


#---------------------------------------------------- GUI DESIGN -------------------------------------------------------

io1 = gr.Interface(
    fn=upload_and_return_prediction_filepath,
    inputs=gr.File(
        label="Upload license plate image(s)",
        type="filepath",
        file_count='multiple',
        height=710,
    ),
    outputs=gr.Textbox(
        label="Detected Text",
        lines = 34,
    ),
    title="License Plate Recognition"
)

io2 = gr.Interface(
    fn=upload_and_return_prediction_image,
    inputs= gr.Image(
        label="Upload license plate image",
        type="filepath",
    ),
    outputs = [
        gr.Textbox(
            label= "Detected Text" ,
    ),
        gr.Image(
            type = 'pil',
            label = 'Detected Images'
        )
    ],
    title="License Plate Recognition"
)

demo = gr.TabbedInterface(
    [io1, io2], ['Mass file prediction', 'Single plate detection']
)

# with gr.Blocks(title="License Plate Recognition") as demo:
#     gr.Markdown("## License Plate Recognition")
#
#     # First Row: File uploader
#     with gr.Row():
#         with gr.Column(elem_id="scroll-upload"):
#             file_input = gr.File(
#                 label="Upload license plate image(s)",
#                 type="filepath",
#                 file_count="multiple",
#                 lines = 10,
#             )
#         with gr.Column():
#             output_box = gr.Textbox(label="Detected Text", height=500)
#
#     # Wire input to output
#     file_input.change(upload_and_return_prediction, inputs=file_input, outputs=output_box)


#---------------------------------------------------- DEPLOYMENT -------------------------------------------------------
# if __name__ == "__main__":
#     demo.launch(css=css_fileLink)

if __name__ == "__main__":
    demo.launch()

