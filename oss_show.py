from flask import Flask, jsonify
from utils.oss import OssManager

app = Flask('oss')
om = OssManager()

@app.route('/oss/')
def index():
    def content(parent_path):
        return {
            'Compare': om.get_compare(parent_path),
            'Winrate': om.get_win_rate(parent_path)
        }

    return jsonify({
        'alpha_zero': content('alpha_zero')
    })

def main():
    app.run(host='0.0.0.0', port=8008)

if __name__ == '__main__':
    main()
