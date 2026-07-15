import requests
from django.shortcuts import render


API_URL = "http://127.0.0.1:8001/predict"

FIELD_SPECS = {
    "runtime_min": {"type": float, "label": "Runtime min"},
    "year": {"type": int, "label": "Year"},
    "age_years": {"type": int, "label": "Age years"},
    "is_animated": {"type": int, "label": "Is animated", "choices": {"0", "1"}},
    "is_tv_series": {"type": int, "label": "Is TV series", "choices": {"0", "1"}},
    "budget_log": {"type": float, "label": "Budget log"},
    "revenue_log": {"type": float, "label": "Revenue log"},
    "box_office_log": {"type": float, "label": "Box office log"},
    "imdb_votes": {"type": float, "label": "IMDb votes"},
    "tmdb_votes": {"type": float, "label": "TMDB votes"},
    "popularity": {"type": float, "label": "Popularity"},
    "IMDb": {"type": float, "label": "IMDb rating"},
    "Metacritic": {"type": float, "label": "Metacritic"},
    "Rotten_Tomatoes": {"type": float, "label": "Rotten Tomatoes"},
    "TMDB": {"type": float, "label": "TMDB rating"},
}

DEFAULT_VALUES = {name: "" for name in FIELD_SPECS}


def parse_field(request_data, field_name):
    raw_value = request_data.get(field_name, "").strip()
    spec = FIELD_SPECS[field_name]

    if raw_value == "":
        return None, f"{spec['label']} is required."

    if "choices" in spec and raw_value not in spec["choices"]:
        return None, f"{spec['label']} must be 0 or 1."

    try:
        return spec["type"](raw_value), None
    except (TypeError, ValueError):
        return None, f"{spec['label']} must be a valid number."


def build_api_payload(parsed_values):
    return {
        "runtime_min": parsed_values["runtime_min"],
        "year": parsed_values["year"],
        "age_years": parsed_values["age_years"],
        "is_animated": parsed_values["is_animated"],
        "is_tv_series": parsed_values["is_tv_series"],
        "budget_log": parsed_values["budget_log"],
        "revenue_log": parsed_values["revenue_log"],
        "box_office_log": parsed_values["box_office_log"],
        "imdb_votes": parsed_values["imdb_votes"],
        "tmdb_votes": parsed_values["tmdb_votes"],
        "popularity": parsed_values["popularity"],
        "IMDb": parsed_values["IMDb"],
        "Metacritic": parsed_values["Metacritic"],
        "Rotten_Tomatoes": parsed_values["Rotten_Tomatoes"],
        "TMDB": parsed_values["TMDB"],
    }


def main(request):
    result = None
    probability_percent = None
    error = None
    field_errors = {}
    values = DEFAULT_VALUES.copy()

    if request.method == "POST":
        parsed_values = {}

        for field_name in FIELD_SPECS:
            values[field_name] = request.POST.get(field_name, "").strip()
            parsed_value, field_error = parse_field(request.POST, field_name)
            if field_error:
                field_errors[field_name] = field_error
            else:
                parsed_values[field_name] = parsed_value

        if field_errors:
            error = "Please fill in all required fields correctly."
        else:
            try:
                response = requests.post(
                    API_URL,
                    json=build_api_payload(parsed_values),
                    timeout=15,
                )
                if response.status_code >= 400:
                    try:
                        api_error = response.json().get("detail", response.text)
                    except ValueError:
                        api_error = response.text
                    raise requests.HTTPError(
                        f"{response.status_code} {response.reason}: {api_error}",
                        response=response,
                    )
                response.raise_for_status()
                data = response.json()

                result = int(data.get("is_mcu_canon", 0))
                probability_percent = round(float(data.get("probability", 0)) * 100, 2)

                values = DEFAULT_VALUES.copy()

            except requests.RequestException as exc:
                error = f"API request failed: {exc}"
            except ValueError:
                error = "API returned invalid JSON."
            except (TypeError, KeyError) as exc:
                error = f"API response is missing required data: {exc}"
            except Exception as exc:
                error = f"Prediction failed: {exc}"

    return render(
        request,
        "main.html",
        {
            "result": result,
            "probability_percent": probability_percent,
            "error": error,
            "field_errors": field_errors,
            "values": values,
        },
    )
