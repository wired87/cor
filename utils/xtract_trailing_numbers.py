def extract_trailing_numbers(text):
    try:
        # 1. Prüfen, ob der String überhaupt mit einer Zahl endet
        if not text or not text[-1].isdigit():
            return ""  # Oder None, falls gewünscht

        # 2. Alle numerischen Zeichen vom Ende her sammeln
        result = []
        for char in reversed(text):
            if char.isdigit():
                result.append(char)
            else:
                # Sobald ein nicht-numerisches Zeichen kommt: Stop
                break

        # Liste umdrehen und zum String zusammenfügen
        return "".join(reversed(result))
    except Exception as e:
        print(f"Err core.utils.xtract_trailing_numbers::extract_trailing_numbers | handler_line=18 | {type(e).__name__}: {e}")
        print(f"[exception] core.utils.xtract_trailing_numbers.extract_trailing_numbers: {e}")
        print("Err extract_trailing_numbers", e)


