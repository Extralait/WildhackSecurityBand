import json

from bottle import run, request
import bottle

from Back.Parser.parser import search_tags

application = bottle.app()

@application.route("/api", method=['OPTIONS', 'POST'])
def api():
    text = dict(request.forms.decode('utf-8'))['text']
    print(text)
    return json.dumps(search_tags(text),indent=4, ensure_ascii=False)


if __name__ == '__main__':
    run(host='localhost', port=8080, debug=True)

