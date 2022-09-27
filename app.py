from server import app, server
from layout import layout

app.layout = layout

if __name__ == "__main__":
    app.run_server(port=8080, debug=True)

