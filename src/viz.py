"""Affichage d'une phrase découpée en tokens, un token par couleur (pour Jupyter / Colab)."""

COLORS = ["#fde2e4", "#e2ece9", "#dfe7fd", "#fff1c1", "#e8dff5", "#d7f2f5"]


def show_tokens(text, tokenizers, names=None):
    """Affiche `text` découpé par chaque tokenizer du dictionnaire `tokenizers` {nom: tokenizer}.

    Un « � » signale un token qui n'est qu'un morceau de lettre (octets bruts).
    """
    from IPython.display import HTML, display

    names = names or list(tokenizers)
    rtl = any("؀" <= c <= "ۿ" for c in text)
    rows = []
    for name in names:
        tok = tokenizers[name]
        ids = tok.encode(text, add_special_tokens=False)
        spans = "".join(
            f'<span style="background:{COLORS[k % len(COLORS)]};padding:2px 1px;margin:0 1px;border-radius:3px">'
            f'{tok.decode([i]).replace(" ", "&nbsp;")}</span>'
            for k, i in enumerate(ids)
        )
        rows.append(
            f'<tr><td style="padding:4px 12px 4px 0;white-space:nowrap"><b>{name}</b><br>{len(ids)} tokens</td>'
            f'<td dir="{"rtl" if rtl else "ltr"}" style="font-size:18px;line-height:2">{spans}</td></tr>'
        )
    display(HTML(f"<table>{''.join(rows)}</table>"))
