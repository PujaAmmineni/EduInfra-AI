# Install requirements (no --user flag)
pip install --no-cache-dir -r requirements.txt
pip install bcrypt flask-cors python-dotenv pillow

# Set PYTHONPATH just in case
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Start the app
exec gunicorn main:app --bind=0.0.0.0:$PORT