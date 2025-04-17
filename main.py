from app import app

if __name__ == "__main__":
    # Debug-Modus aktivieren und Server auf 0.0.0.0 (alle Interfaces) hören lassen
    app.run(host="0.0.0.0", port=5000, debug=True)
