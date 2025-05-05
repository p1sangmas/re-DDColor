from app import Flask, request, send_file
import subprocess

app = Flask(__name__)

@app.route("/colorize", methods=["POST"])
def colorize():
    # Save uploaded file
    uploaded_file = request.files["file"]
    input_path = "uploaded_image.JPEG"
    uploaded_file.save(input_path)

    # Run the colorization script
    output_path = "colorized_image.png"
    command = [
        "CUDA_VISIBLE_DEVICES=0",
        "python3", "inference/colorization_pipeline.py",
        "--input_file", input_path,
        "--output_file", output_path,
        "--model_path", "modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt"
    ]
    subprocess.run(" ".join(command), shell=True)

    # Return the colorized image
    return send_file(output_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
