import csv
from datetime import datetime


def load_prices_csv(
    path,
    price_col="Close",
    date_col=None,
    max_rows=None,
):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if price_col not in row:
                raise ValueError(
                    f"Missing column '{price_col}' in {path}"
                )
            try:
                price = float(row[price_col])
            except (TypeError, ValueError):
                continue
            if date_col:
                try:
                    date_val = datetime.fromisoformat(row[date_col])
                except (TypeError, ValueError):
                    date_val = row[date_col]
                rows.append((date_val, price))
            else:
                rows.append(price)
            if max_rows and len(rows) >= max_rows:
                break

    if date_col:
        rows.sort(key=lambda item: item[0])
        return [price for _, price in rows]
    return rows
