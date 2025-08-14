from alpha_blokus.__main__ import main

def test_selfplay_loop():
    main({
        "entrypoint": "selfplay_loop",
    })
