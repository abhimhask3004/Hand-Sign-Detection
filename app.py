from flask import Flask, render_template, jsonify
from sample import generate_frames # Import your computer vision function
app = Flask(__name__)

@app.route("/")
def index():
   return render_template('index.html')
   #return "Hello this is a main page <h1>hello</h1>"

@app.route('/api/process_video', methods=['GET'])
def api_process_video():
   output = generate_frames()  # Call your computer vision function
   return jsonify({'result': output})

if __name__ == "__main__":
    app.debug=True
    app.run()