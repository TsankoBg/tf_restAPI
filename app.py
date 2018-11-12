
import io
from flask import Flask, render_template, request

from PIL import Image
from flask import send_file

app = Flask(__name__)



# detector.detectNumberPlate('twocar.jpg')


@app.route("/")
def index():
    return render_template('index.html')





if __name__ == "__main__":
    app.run()
