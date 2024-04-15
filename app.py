from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from io import BytesIO
import base64


from data_processing.ergm import run_ergm

from data_processing.alaam import run_alaam

# app.py
from graph_utils import generate_network_graph


app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

def allowed_file(filename):
    """Check file extension to be one of the allowed types."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page with the upload form."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the uploading of files."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('File successfully uploaded')
        return redirect(url_for('uploaded_file', filename=filename))
    else:
        flash('Allowed file types are txt, csv, xlsx')
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve the uploaded files from the upload folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/show-stats')
def show_stats():
    # You might calculate stats or get them from somewhere
    stats_data = {
        "mean": 100,
        "median": 50,
        "mode": 25
    }
    # Pass the stats_data to the HTML template
    return render_template('stats_output.html', stats=stats_data)


@app.route('/process-data', methods=['POST'])
def process_data():
    # Assume file is saved in 'uploads/' and filename is received via POST
    filename = request.form['filename']
    file_path = f'uploads/{filename}'

    # Process data with ERGM
    ergm_result = run_ergm(file_path)

    # Optionally, process with ALAAM
    alaam_result = run_alaam(file_path)

    # Store results or pass to template for displaying
    return render_template('results.html', ergm_result= ergm_result, alaam_result=alaam_result)

@app.route('/show-network')
def show_network():
    # Generate a network graph
    fig = generate_network_graph()
    
    # Convert plot to PNG image
    img = BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('visualization.html', plot_url=plot_url)


if __name__ == "__main__":
    app.run(debug=True)




