import requests
import os
import re
import pandas as pd
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


load_dotenv()
API_KEY = os.getenv("API_KEY")


def extraer_steamid(entrada):
    """
    Detecta automáticamente si el usuario ingresó:
    - SteamID numérico
    - URL con /profiles/
    - URL con /id/
    - Alias directo
    """

    if not entrada:
        logging.error("Entrada vacía")
        return None

    entrada = str(entrada).strip()

    # 1️⃣ SteamID64 directo
    if entrada.isdigit() and len(entrada) == 17:
        logging.info("SteamID numérico detectado")
        return entrada

    # 2️⃣ URL tipo /profiles/
    match_profile = re.search(r"/profiles/(\d+)", entrada)
    if match_profile:
        logging.info("URL con SteamID detectada")
        return match_profile.group(1)

    # 3️⃣ URL tipo /id/alias
    match_vanity = re.search(r"/id/([^/?]+)", entrada)
    if match_vanity:
        vanity = match_vanity.group(1).lower()

        if len(vanity) < 3:
            logging.error("Alias demasiado corto")
            return None

        logging.info(f"Alias detectado en URL: {vanity}")
        return resolver_vanity(vanity)

    if "/" not in entrada:

        alias = entrada.lower()

        if len(alias) < 3:
            logging.error("Alias demasiado corto")
            return None

        logging.info(f"Alias directo detectado: {alias}")
        return resolver_vanity(alias)

    logging.error("No se pudo interpretar la entrada")
    return None



cache_vanity = {}

def resolver_vanity(vanity):
    """
    Convierte un alias de Steam a SteamID64 usando la API.
    Usa cache para evitar llamadas repetidas.
    """

    vanity = vanity.lower()

    # comprobar cache
    if vanity in cache_vanity:
        logging.info("Alias encontrado en cache")
        return cache_vanity[vanity]

    url = "https://api.steampowered.com/ISteamUser/ResolveVanityURL/v1/"

    params = {
        "key": API_KEY,
        "vanityurl": vanity
    }

    try:
        logging.info(f"Resolviendo alias: {vanity}")

        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            logging.error(f"Error HTTP: {response.status_code}")
            return None

        data = response.json()

        if "response" not in data:
            logging.error("Respuesta inválida de Steam")
            return None

        if data["response"].get("success") == 1:

            steamid = data["response"].get("steamid")

            logging.info(f"Alias resuelto: {steamid}")

            # guardar en cache
            cache_vanity[vanity] = steamid

            return steamid

        logging.error("Alias no encontrado o inválido")
        return None

    except requests.exceptions.Timeout:
        logging.error("Tiempo de espera agotado")
        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error de conexión: {e}")
        return None

    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return None
        
    
def obtener_juegos_usuario(steamid):

    logging.info(f"Consultando juegos del usuario {steamid}")

    url = "https://api.steampowered.com/IPlayerService/GetOwnedGames/v1/"

    params = {
        "key": API_KEY,
        "steamid": steamid,
        "include_appinfo": True,
        "include_played_free_games": True
    }

    try:

        logging.info("Enviando petición a la API de Steam")

        response = requests.get(url, params=params, timeout=10)

        logging.info(f"Código de respuesta: {response.status_code}")

        if response.status_code != 200:
            logging.error("La API devolvió un error")
            return pd.DataFrame()

        data = response.json()

        juegos = data.get("response", {}).get("games", [])

        logging.info(f"Se encontraron {len(juegos)} juegos")

        df = pd.DataFrame(juegos)

        return df

    except requests.exceptions.Timeout:
        logging.error("Tiempo de espera agotado al consultar juegos")
        return pd.DataFrame()

    except requests.RequestException as e:
        logging.error(f"Error en la petición: {e}")
        return pd.DataFrame()

    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return pd.DataFrame()

def construir_perfil_usuario(df, juegos_usuario, min_minutos=120):

    if juegos_usuario is None or juegos_usuario.empty:
        logging.warning("El usuario no tiene juegos o el perfil es privado")
        return df.iloc[0:0]

    # Filtrar juegos con al menos X minutos jugados
    juegos_filtrados = juegos_usuario[
        juegos_usuario["playtime_forever"] >= min_minutos
    ]

    logging.info(f"Juegos con al menos {min_minutos} minutos jugados: {len(juegos_filtrados)}")

    if juegos_filtrados.empty:
        logging.warning("Ningún juego supera el mínimo de tiempo jugado")
        return df.iloc[0:0]

    # Obtener appids
    appids_usuario = set(juegos_filtrados["appid"].astype(str))

    logging.info(f"AppIDs válidos: {len(appids_usuario)}")

    if "app_id" not in df.columns:
        logging.error("El DataFrame no contiene la columna 'app_id'")
        return df.iloc[0:0]

    # Filtrar dataset
    juegos_df = df[df["app_id"].isin(appids_usuario)]

    logging.info(f"Juegos encontrados en dataset: {len(juegos_df)}")

    return juegos_df