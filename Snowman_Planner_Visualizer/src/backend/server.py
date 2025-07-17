from flask import Flask, send_file, jsonify, make_response
import os
import glob
import logging

# Set the working directory to the project root (snowman-planner)
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

app = Flask(__name__, static_folder='src/frontend/static', template_folder='src/frontend/templates')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def serve_index():
    try:
        index_path = os.path.abspath(os.path.join('src', 'frontend', 'templates', 'index.html'))
        if not os.path.exists(index_path):
            logger.error(f"index.html not found at {index_path}")
            return make_response(jsonify({"error": f"index.html not found at {index_path}"}), 404)
        return send_file(index_path)
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return make_response(jsonify({"error": f"Error serving index.html: {str(e)}"}), 500)

@app.route('/plans/<path:filename>')
def serve_plan(filename):
    try:
        file_path = os.path.join('plans', filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            logger.error(f"Plan file {filename} not found")
            return make_response(jsonify({"error": f"Plan file {filename} not found in plans directory"}), 404)
    except Exception as e:
        logger.error(f"Error serving plan file {filename}: {str(e)}")
        return make_response(jsonify({"error": f"Error serving plan file: {str(e)}"}), 500)

@app.route('/plans')
def list_plans():
    try:
        plans = glob.glob('plans/*.txt')
        return jsonify([os.path.basename(p) for p in plans])
    except Exception as e:
        logger.error(f"Error listing plans: {str(e)}")
        return make_response(jsonify({"error": f"Error listing plans: {str(e)}"}), 500)

if __name__ == '__main__':
    logger.info(f"Current working directory: {os.getcwd()}")
    app.run(host='0.0.0.0', port=8000, debug=False)
