# src/preprocessing.py

import pandas as pd
from .config import TRAIN_START_DATE, TRAIN_END_DATE, DATA_PROCESSED_DIR


VALID_FIFA_TEAMS = {
    'Abkhazia', 'Afghanistan', 'Albania', 'Alderney', 'Algeria', 'Ambazonia',
    'American Samoa', 'Andalusia', 'Andorra', 'Angola', 'Anguilla',
    'Antigua and Barbuda', 'Arameans Suryoye', 'Argentina', 'Armenia',
    'Artsakh', 'Aruba', 'Asturias', 'Australia', 'Austria', 'Aymara',
    'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barawa', 'Barbados',
    'Basque Country', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bermuda',
    'Bhutan', 'Biafra', 'Bolivia', 'Bonaire', 'Bosnia and Herzegovina',
    'Botswana', 'Brazil', 'British Virgin Islands', 'Brittany', 'Brunei',
    'Bulgaria', 'Burkina Faso', 'Burundi', 'Cambodia', 'Cameroon', 'Canada',
    'Canary Islands', 'Cape Verde', 'Cascadia', 'Catalonia', 'Cayman Islands',
    'Central African Republic', 'Central Spain', 'Chad', 'Chagos Islands',
    'Chameria', 'Chechnya', 'Chile', 'China PR', 'Cilento', 'Colombia',
    'Comoros', 'Congo', 'Cook Islands', 'Corsica', 'Costa Rica',
    'County of Nice', 'Crimea', 'Croatia', 'Cuba', 'Curaçao', 'Cyprus',
    'Czech Republic', 'Czechoslovakia', 'DR Congo', 'Darfur', 'Denmark',
    'Djibouti', 'Dominica', 'Dominican Republic', 'Donetsk PR', 'Délvidék',
    'Ecuador', 'Egypt', 'El Salvador', 'Elba Island', 'Ellan Vannin',
    'England', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini',
    'Ethiopia', 'Falkland Islands', 'Faroe Islands', 'Felvidék', 'Fiji',
    'Finland', 'France', 'Franconia', 'French Guiana', 'Frøya', 'Gabon',
    'Galicia', 'Gambia', 'Georgia', 'German DR', 'Germany', 'Ghana',
    'Gibraltar', 'Gotland', 'Gozo', 'Greece', 'Greenland', 'Grenada',
    'Guadeloupe', 'Guam', 'Guatemala', 'Guernsey', 'Guinea', 'Guinea-Bissau',
    'Guyana', 'Găgăuzia', 'Haiti', 'Hitra', 'Hmong', 'Honduras', 'Hong Kong',
    'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq',
    'Iraqi Kurdistan', 'Isle of Man', 'Isle of Wight', 'Israel', 'Italy',
    'Ivory Coast', 'Jamaica', 'Japan', 'Jersey', 'Jordan', 'Kabylia',
    'Kazakhstan', 'Kenya', 'Kernow', 'Kiribati', 'Kosovo', 'Kuwait',
    'Kyrgyzstan', 'Kárpátalja', 'Laos', 'Latvia', 'Lebanon', 'Lesotho',
    'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luhansk PR',
    'Luxembourg', 'Macau', 'Madagascar', 'Madrid', 'Malawi', 'Malaysia',
    'Maldives', 'Mali', 'Malta', 'Manchukuo', 'Mapuche', 'Marshall Islands',
    'Martinique', 'Matabeleland', 'Maule Sur', 'Mauritania', 'Mauritius',
    'Mayotte', 'Menorca', 'Mexico', 'Micronesia', 'Moldova', 'Monaco',
    'Mongolia', 'Montenegro', 'Montserrat', 'Morocco', 'Mozambique',
    'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Caledonia',
    'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'North Korea',
    'North Macedonia', 'North Vietnam', 'Northern Cyprus', 'Northern Ireland',
    'Northern Mariana Islands', 'Norway', 'Occitania', 'Oman', 'Orkney',
    'Padania', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Panjab',
    'Papua New Guinea', 'Paraguay', 'Parishes of Jersey', 'Peru',
    'Philippines', 'Poland', 'Portugal', 'Provence', 'Puerto Rico', 'Qatar',
    'Raetia', 'Republic of Ireland', 'Republic of St. Pauli', 'Rhodes',
    'Romani people', 'Romania', 'Russia', 'Rwanda', 'Ryūkyū', 'Réunion',
    'Saare County', 'Saarland', 'Saint Barthélemy', 'Saint Helena',
    'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Martin',
    'Saint Pierre and Miquelon', 'Saint Vincent and the Grenadines', 'Samoa',
    'San Marino', 'Sark', 'Saudi Arabia', 'Saugeais', 'Scotland', 'Sealand',
    'Seborga', 'Senegal', 'Serbia', 'Seychelles', 'Shetland', 'Sierra Leone',
    'Silesia', 'Singapore', 'Sint Maarten', 'Slovakia', 'Slovenia',
    'Solomon Islands', 'Somalia', 'Somaliland', 'South Africa', 'South Korea',
    'South Ossetia', 'South Sudan', 'South Yemen', 'Spain', 'Sri Lanka',
    'Sudan', 'Suriname', 'Surrey', 'Sweden', 'Switzerland', 'Syria',
    'Székely Land', 'Sápmi', 'São Tomé and Príncipe', 'Tahiti', 'Taiwan',
    'Tajikistan', 'Tamil Eelam', 'Tanzania', 'Thailand', 'Tibet', 'Ticino',
    'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia',
    'Turkey', 'Turkmenistan', 'Turks and Caicos Islands', 'Tuvalu',
    'Two Sicilies', 'Uganda', 'Ukraine', 'United Arab Emirates',
    'United Koreans in Japan', 'United States', 'United States Virgin Islands',
    'Uruguay', 'Uzbekistan', 'Vanuatu', 'Vatican City', 'Venezuela',
    'Vietnam', 'Vietnam Republic', 'Wales', 'Wallis Islands and Futuna',
    'West Papua', 'Western Armenia', 'Western Australia', 'Western Isles',
    'Western Sahara', 'Yemen', 'Yemen DPR', 'Ynys Môn', 'Yorkshire',
    'Yoruba Nation', 'Yugoslavia', 'Zambia', 'Zanzibar', 'Zimbabwe', 'Åland Islands'
}


NAME_NORMALIZATION = {
    "Côte d'Ivoire": "Ivory Coast",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
}


WORLD_CUP_RELEVANT_TEAMS = {
    # Group A/B/C/D/E/F/G/H/I/J/K/L assumed participants
    "Mexico",
    "South Africa",
    "South Korea",     # from "Korea Republic"
    "Denmark",
    "Canada",
    "Italy",
    "United States",
    "Paraguay",
    "Haiti",
    "Scotland",
    "Australia",
    "Turkey",
    "Brazil",
    "Morocco",
    "Qatar",
    "Switzerland",
    "Ivory Coast",     # from "Côte d'Ivoire"
    "Ecuador",
    "Germany",
    "Curaçao",
    "Netherlands",
    "Japan",
    "Poland",
    "Tunisia",
    "Saudi Arabia",
    "Uruguay",
    "Spain",
    "Cape Verde",      # from "Cabo Verde"
    "Iran",            # from "IR Iran"
    "New Zealand",
    "Belgium",
    "Egypt",
    "France",
    "Senegal",
    "Bolivia",
    "Norway",
    "Argentina",
    "Algeria",
    "Austria",
    "Jordan",
    "Ghana",
    "Panama",
    "England",
    "Croatia",
    "Portugal",
    "Jamaica",
    "Uzbekistan",
    "Colombia",
}


def filter_to_worldcup_relevant(df):
    """
    Keep only matches where AT LEAST ONE of the teams is in the
    World Cup 2026 relevant set.

    This ensures all relevant teams have enough matches, while still
    discarding games that are completely irrelevant (neither team
    will appear at the World Cup).
    """
    df = df.copy()
    mask = df["home_team"].isin(WORLD_CUP_RELEVANT_TEAMS) | df["away_team"].isin(WORLD_CUP_RELEVANT_TEAMS)
    return df[mask]



def normalize_team_names(df):
    df = df.copy()
    df["home_team"] = df["home_team"].replace(NAME_NORMALIZATION)
    df["away_team"] = df["away_team"].replace(NAME_NORMALIZATION)
    return df


def filter_to_fifa_teams(df):
    return df[
        (df["home_team"].isin(VALID_FIFA_TEAMS)) &
        (df["away_team"].isin(VALID_FIFA_TEAMS))
    ].copy()


def apply_former_name_mapping(results: pd.DataFrame, former_names: pd.DataFrame | None) -> pd.DataFrame:
    """
    Map any historical team names to their 'current' names using former_names.csv.
    For 2018–2025 this might barely matter, but it's nice to have.
    """
    if former_names is None:
        return results

    # Build mapping dict: former -> current
    mapping = dict(zip(former_names["former"], former_names["current"]))

    results = results.copy()
    results["home_team"] = results["home_team"].replace(mapping)
    results["away_team"] = results["away_team"].replace(mapping)
    return results


def filter_by_date(results: pd.DataFrame) -> pd.DataFrame:
    mask = (results["date"] >= TRAIN_START_DATE) & (results["date"] <= TRAIN_END_DATE)
    return results.loc[mask].copy()


def select_relevant_columns(results: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]
    # Only keep columns that actually exist (defensive programming)
    keep_cols = [c for c in keep_cols if c in results.columns]
    return results[keep_cols].copy()


def preprocess_matches(results, former_names):
    # 1) historical name mapping
    df = apply_former_name_mapping(results, former_names)

    # 2) normalize modern variants (IR Iran -> Iran, Korea Republic -> South Korea, etc.)
    df = normalize_team_names(df)

    # 3) date filter (2018–2025, as you had before)
    df = filter_by_date(df)

    # 4) keep only relevant columns
    df = select_relevant_columns(df)

    # 5) restrict to World Cup relevant teams ONLY
    df = filter_to_worldcup_relevant(df)

    # 6) enforce types
    df["neutral"] = df["neutral"].astype(bool)

    return df





def save_processed_matches(df: pd.DataFrame, filename: str = "matches_2018_2025.csv") -> None:
    output_path = f"{DATA_PROCESSED_DIR}/{filename}"
    df.to_csv(output_path, index=False)
    print(f"Saved processed matches to {output_path}")
