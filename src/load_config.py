import json
import logging
import os
import tempfile
from pathlib import Path
from logging.handlers import RotatingFileHandler
from threading import RLock
from typing import Optional, Dict

# ---- Project-wide defaults (override via env) ----
DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FILE  = os.getenv("LOG_FILE", "logs/app.log")

# Module-level logger setup
_logger_initialized = False
_init_lock = RLock()

# Root/module-level logger (safe to use after setup_logging())
logger = logging.getLogger(__name__)

def _coerce_level(level_str: str) -> int:
    levels = {
        "DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING,
        "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL,
    }
    return levels.get((level_str or "INFO").upper(), logging.INFO)

def _ensure_dir_for(path_str: str) -> bool:
    try:
        p = Path(path_str).expanduser()
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def _make_file_handler(target_path: str, level: int) -> logging.Handler | None:
    try:
        # Rotating handler avoids unbounded file growth; tweak as needed.
        fh = RotatingFileHandler(
            filename=target_path,
            mode="a",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
            delay=True,
        )
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        return fh
    except Exception:
        return None

def _make_console_handler(level: int) -> logging.Handler | None:
    try:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        return ch
    except Exception:
        return None

def setup_logging(log_file: str = DEFAULT_LOG_FILE, level: str | None = None) -> logging.Logger:
    """
    Best-effort, exception-safe logger setup.
    - Creates directory for log_file if possible.
    - Tries file handler -> falls back to console -> falls back to NullHandler.
    - Idempotent and thread-safe.
    """
    global _logger_initialized
    if _logger_initialized:
        return logging.getLogger()  # return root (already configured)

    with _init_lock:
        if _logger_initialized:
            return logging.getLogger()

        try:
            log_level = _coerce_level(level or DEFAULT_LOG_LEVEL)
            root = logging.getLogger()
            root.setLevel(log_level)

            # Clear once to avoid duplicate handlers on re-imports or reloads
            root.handlers[:] = []

            # Try file handler at requested path
            file_handler = None
            if _ensure_dir_for(log_file):
                file_handler = _make_file_handler(log_file, log_level)

            # If that failed, try a simple file in CWD
            if file_handler is None:
                fallback_path = "app.log"
                if _ensure_dir_for(fallback_path):
                    file_handler = _make_file_handler(fallback_path, log_level)

            # If still failed, try temp directory
            if file_handler is None:
                tmp_path = str(Path(tempfile.gettempdir()) / "app.log")
                if _ensure_dir_for(tmp_path):
                    file_handler = _make_file_handler(tmp_path, log_level)

            if file_handler:
                root.addHandler(file_handler)

            # Add console as well (best effort)
            console_handler = _make_console_handler(log_level)
            if console_handler:
                root.addHandler(console_handler)

            # If nothing worked, attach a NullHandler so logging calls never explode
            if not root.handlers:
                root.addHandler(logging.NullHandler())

            # Avoid propagation surprises for named loggers
            root.propagate = False

            _logger_initialized = True
            root.info("Logging initialized (level=%s)", logging.getLevelName(log_level))
            return root
        except Exception:
            # As an absolute last resort, ensure logging never breaks app code
            bare = logging.getLogger()
            bare.handlers[:] = [logging.NullHandler()]
            bare.setLevel(logging.INFO)
            _logger_initialized = True
            return bare

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a logger that's safe to use anywhere.
    - Ensures global logging is configured.
    - Returns a child logger if `name` provided; else returns a module-level logger.
    """
    setup_logging()
    return logging.getLogger(name) if name else logging.getLogger(__name__)

def set_global_log_level(level: str) -> None:
    """
    Dynamically change global log level (never raises).
    """
    try:
        lvl = _coerce_level(level)
        root = logging.getLogger()
        root.setLevel(lvl)
        for h in root.handlers:
            try:
                h.setLevel(lvl)
            except Exception:
                continue
        root.info("Global log level set to %s", logging.getLevelName(lvl))
    except Exception:
        # swallow
        pass

def validate_comma_separated_list(value, field_name):
    """Validate that a value is a string and either empty or a valid comma-separated list."""
    if not isinstance(value, str):
        logger.error(f"Invalid type for {field_name}: expected string, got {type(value).__name__}")
        raise TypeError(f"Invalid type for {field_name}: expected string, got {type(value).__name__}")
    if value and not all(part.strip() for part in value.split(',')):
        logger.error(f"Invalid format for {field_name}: contains empty or whitespace-only entries")
        raise ValueError(f"Invalid format for {field_name}: contains empty or whitespace-only entries")
    logger.debug(f"Validated {field_name}: {value}")
    return value

def get_mssql_connection_string(sql_params):
    """
    Construct a pyodbc connection string from SQL connection parameters.
    """
    logger.debug("Constructing MSSQL connection string")
    trust_cert = "yes" if sql_params['trust_cert'] else "no"
    conn_string = (
        f"DRIVER={{{sql_params['driver']}}};"
        f"SERVER={sql_params['server']};"
        f"DATABASE={sql_params['database']};"
        f"UID={sql_params['uid']};"
        f"PWD={sql_params['pwd']};"
        f"TrustServerCertificate={trust_cert}"
    )
    logger.debug(f"Constructed MSSQL connection string: {conn_string}")
    return conn_string

def load_config(config_path="config/config.json"):
    """
    Load configuration from a JSON file and return both config and logger.
    Returns a tuple (config_dict, logger_instance).
    If config file is missing, returns (None, logger_instance).
    Raises appropriate errors for invalid JSON or missing required keys.
    """
    # Initialize logging first and get the logger
    logger = setup_logging()
    logger.info(f"Attempting to load configuration from {config_path}")
    
    # Check if config file exists
    if not os.path.isfile(config_path):
        logger.warning(f"Config file not found at: {config_path}. Returning None config with logger.")
        return None, logger

    try:
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
        logger.debug("Configuration loaded successfully")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {config_path}: {str(e)}")
        raise ValueError(f"Invalid JSON format in {config_path}: {str(e)}")

    # Validate required top-level keys based on actual config.json structure
    required_keys = [
        'sql_connection', 'sql_processing', 'table_relations', 'flat_file_data',
        'domain_mapping', 'tables_dir', 'debug', 'query', 'embeddings', 'top_k',
        'generated_embedding_store', 'domain_mapped_csv_store', 'company_mapped_data',
        'text_cols', 'min_score', 'exact_only', 'prefer_exact',
        'boost_exact', 'columns', 'show_source', 'sentence_transformer_model_name', 
        'sentence_transformer_model_from_net', 'batch_size', 'qwen_model_path'
    ]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        logger.error(f"Missing required keys in config.json: {', '.join(missing_keys)}")
        raise KeyError(f"Missing required keys in config.json: {', '.join(missing_keys)}")

    # Validate SQL connection parameters
    required_sql_keys = ['server', 'database', 'uid', 'pwd', 'driver', 'trust_cert']
    missing_sql_keys = [key for key in required_sql_keys if key not in config['sql_connection']]
    if missing_sql_keys:
        logger.error(f"Missing required SQL connection keys in config.json: {', '.join(missing_sql_keys)}")
        raise KeyError(f"Missing required SQL connection keys in config.json: {', '.join(missing_sql_keys)}")

    # Validate SQL connection parameter types
    for key in ['server', 'database', 'uid', 'pwd', 'driver']:
        if not isinstance(config['sql_connection'][key], str):
            logger.error(f"Invalid type for sql_connection.{key}: expected string, got {type(config['sql_connection'][key]).__name__}")
            raise TypeError(f"Invalid type for sql_connection.{key}: expected string, got {type(config['sql_connection'][key]).__name__}")
    if not isinstance(config['sql_connection']['trust_cert'], bool):
        logger.error(f"Invalid type for sql_connection.trust_cert: expected boolean, got {type(config['sql_connection']['trust_cert']).__name__}")
        raise TypeError(f"Invalid type for sql_connection.trust_cert: expected boolean, got {type(config['sql_connection']['trust_cert']).__name__}")
    logger.debug("SQL connection parameters validated successfully")

    # Validate SQL processing parameters
    required_sql_processing_keys = [
        'table_relations', 'flat_file_data', 'include_schemas', 'exclude_schemas',
        'include_tables', 'exclude_tables', 'max_rows', 'batch_size'
    ]
    missing_sql_processing_keys = [key for key in required_sql_processing_keys if key not in config['sql_processing']]
    if missing_sql_processing_keys:
        logger.error(f"Missing required SQL processing keys in config.json: {', '.join(missing_sql_processing_keys)}")
        raise KeyError(f"Missing required SQL processing keys in config.json: {', '.join(missing_sql_processing_keys)}")

    # Validate SQL processing parameter types and values
    for key in ['table_relations', 'flat_file_data']:
        if not isinstance(config['sql_processing'][key], str):
            logger.error(f"Invalid type for sql_processing.{key}: expected string, got {type(config['sql_processing'][key]).__name__}")
            raise TypeError(f"Invalid type for sql_processing.{key}: expected string, got {type(config['sql_processing'][key]).__name__}")
    for key in ['include_schemas', 'exclude_schemas', 'include_tables', 'exclude_tables']:
        validate_comma_separated_list(config['sql_processing'][key], f"sql_processing.{key}")
    if config['sql_processing']['max_rows'] is not None and (not isinstance(config['sql_processing']['max_rows'], int) or config['sql_processing']['max_rows'] <= 0):
        logger.error(f"Invalid value for sql_processing.max_rows: expected null or positive integer, got {config['sql_processing']['max_rows']}")
        raise ValueError(f"Invalid value for sql_processing.max_rows: expected null or positive integer, got {config['sql_processing']['max_rows']}")
    if not isinstance(config['sql_processing']['batch_size'], int) or config['sql_processing']['batch_size'] <= 0:
        logger.error(f"Invalid value for sql_processing.batch_size: expected positive integer, got {config['sql_processing']['batch_size']}")
        raise ValueError(f"Invalid value for sql_processing.batch_size: expected positive integer, got {config['sql_processing']['batch_size']}")
    logger.debug("SQL processing parameters validated successfully")

    # Validate top-level parameters
    for key in ['table_relations', 'flat_file_data', 'domain_mapping', 'generated_embedding_store', 'domain_mapped_csv_store',
                'sentence_transformer_model_name', 'sentence_transformer_model_from_net', 'qwen_model_path']:
        if not isinstance(config[key], str):
            logger.error(f"Invalid type for {key}: expected string, got {type(config[key]).__name__}")
            raise TypeError(f"Invalid type for {key}: expected string, got {type(config[key]).__name__}")
    if config['tables_dir'] is not None and not isinstance(config['tables_dir'], str):
        logger.error(f"Invalid type for tables_dir: expected string or null, got {type(config['tables_dir']).__name__}")
        raise TypeError(f"Invalid type for tables_dir: expected string or null, got {type(config['tables_dir']).__name__}")
    if not isinstance(config['debug'], bool):
        logger.error(f"Invalid type for debug: expected boolean, got {type(config['debug']).__name__}")
        raise TypeError(f"Invalid type for debug: expected boolean, got {type(config['debug']).__name__}")
    if not isinstance(config['batch_size'], int) or config['batch_size'] <= 0:
        logger.error(f"Invalid value for batch_size: expected positive integer, got {config['batch_size']}")
        raise ValueError(f"Invalid value for batch_size: expected positive integer, got {config['batch_size']}")
    for key in ['query', 'embeddings', 'text_cols', 'columns']:
        if config[key] is not None and not isinstance(config[key], str):
            logger.error(f"Invalid type for {key}: expected string or null, got {type(config[key]).__name__}")
            raise TypeError(f"Invalid type for {key}: expected string or null, got {type(config[key]).__name__}")
    if not isinstance(config['top_k'], int) or config['top_k'] <= 0:
        logger.error(f"Invalid value for top_k: expected positive integer, got {config['top_k']}")
        raise ValueError(f"Invalid value for top_k: expected positive integer, got {config['top_k']}")
    if config['min_score'] is not None and not isinstance(config['min_score'], (int, float)):
        logger.error(f"Invalid type for min_score: expected number or null, got {type(config['min_score']).__name__}")
        raise TypeError(f"Invalid type for min_score: expected number or null, got {type(config['min_score']).__name__}")
    for key in ['exact_only', 'prefer_exact', 'show_source']:
        if not isinstance(config[key], bool):
            logger.error(f"Invalid type for {key}: expected boolean, got {type(config[key]).__name__}")
            raise TypeError(f"Invalid type for {key}: expected boolean, got {type(config[key]).__name__}")
    if not isinstance(config['boost_exact'], (int, float)):
        logger.error(f"Invalid type for boost_exact: expected number, got {type(config['boost_exact']).__name__}")
        raise TypeError(f"Invalid type for boost_exact: expected number, got {type(config['boost_exact']).__name__}")
    for key in ['text_cols', 'columns']:
        if config[key]:
            validate_comma_separated_list(config[key], key)
    logger.debug("Validated columns: {config.get('columns', 'None')}")
    logger.debug("Top-level parameters validated successfully")

    # Validate company_mapped_data structure
    if not isinstance(config['company_mapped_data'], dict):
        logger.error("Invalid type for company_mapped_data: expected dict, got {type(config['company_mapped_data']).__name__}")
        raise TypeError("Invalid type for company_mapped_data: expected dict, got {type(config['company_mapped_data']).__name__}")
    
    required_company_mapped_keys = ['processed_data_store', 'tfidf_search_store', 'dense_index_store']
    missing_company_mapped_keys = [key for key in required_company_mapped_keys if key not in config['company_mapped_data']]
    if missing_company_mapped_keys:
        logger.error(f"Missing required company_mapped_data keys in config.json: {', '.join(missing_company_mapped_keys)}")
        raise KeyError(f"Missing required company_mapped_data keys in config.json: {', '.join(missing_company_mapped_keys)}")
    
    for key in required_company_mapped_keys:
        if not isinstance(config['company_mapped_data'][key], str):
            logger.error(f"Invalid type for company_mapped_data.{key}: expected string, got {type(config['company_mapped_data'][key]).__name__}")
            raise TypeError(f"Invalid type for company_mapped_data.{key}: expected string, got {type(config['company_mapped_data'][key]).__name__}")
    logger.debug("Company mapped data structure validated successfully")

    # Convert paths to Path objects
    config['sql_processing']['flat_file_data'] = Path(config['sql_processing']['flat_file_data'])
    config['tables_dir'] = Path(config['tables_dir']) if config['tables_dir'] is not None else None
    config['generated_embedding_store'] = Path(config['generated_embedding_store'])
    config['domain_mapped_csv_store'] = Path(config['domain_mapped_csv_store'])
    config['table_relations'] = Path(config['table_relations'])
    config['flat_file_data'] = Path(config['flat_file_data'])
    config['domain_mapping'] = Path(config['domain_mapping'])
    logger.debug("Converted paths to Path objects")

    # Validate file and directory existence (only for critical paths)
    if not os.path.isfile(config['domain_mapping']):
        logger.error(f"Domain mapping file not found at: {config['domain_mapping']}")
        raise FileNotFoundError(f"Domain mapping file not found at: {config['domain_mapping']}")
    if not os.path.isfile(config['table_relations']):
        logger.error(f"Table relations file not found at: {config['table_relations']}")
        raise FileNotFoundError(f"Table relations file not found at: {config['table_relations']}")
    if config['tables_dir'] is not None and not os.path.isdir(config['tables_dir']):
        logger.error(f"Tables directory not found at: {config['tables_dir']}")
        raise NotADirectoryError(f"Tables directory not found at: {config['tables_dir']}")
    if not os.path.isfile(config['qwen_model_path']):
        logger.error(f"Model file not found at: {config['qwen_model_path']}")
        raise FileNotFoundError(f"Model file not found at: {config['qwen_model_path']}")
    logger.debug("All file and directory paths validated successfully")
    logger.info("Configuration loaded and validated successfully")

    return config, logger

def load_config_only(config_path="config/config.json"):
    """
    Backward compatibility function that returns only the config.
    For new code, prefer load_config() which returns (config, logger).
    """
    (config, _) = load_config(config_path)
    return config

def get_sql_connection_params(config):
    """Extract SQL connection parameters from config"""
    logger.debug("Extracting SQL connection parameters")
    return config['sql_connection'] if config and 'sql_connection' in config else None

def get_sql_processing_params(config):
    """Extract SQL processing parameters from config"""
    logger.debug("Extracting SQL processing parameters")
    return config['sql_processing'] if config and 'sql_processing' in config else None

def get_processing_params(config):
    """Extract processing parameters for CSV generation"""
    logger.debug("Extracting processing parameters")
    print("===================")
    print(config)
    return {
        'domain_mapping': config['domain_mapping'],
        'table_relations': config['table_relations'],
        'tables_dir': config['tables_dir'],
        'outdir': config.get('outdir', config.get('domain_mapped_csv_store')),
        'debug': config['debug']
    } if config else None

def get_embedding_params(config):
    """Extract embedding generation parameters"""
    logger.debug("Extracting embedding parameters")
    return {
        'build_embeddings': config.get('build_embeddings', config.get('domain_mapped_csv_store')),
        'stores_root': config.get('stores_root', config.get('generated_embedding_store', 'embeddings_store')),
        'model_name': config.get('model_name', config.get('sentence_transformer_model_name', 'sentence-transformers/all-mpnet-base-v2')),
        'batch_size': config['batch_size'],
        'qwen_model_path': config['qwen_model_path']
    } if config else None


def get_embedding_store(config: Dict) -> Optional[Path]:
    """Extract embedding generation parameters and return a Path object.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        Path: A Path object representing the embedding store directory, or None if invalid.

    Raises:
        KeyError: If 'generated_embedding_store' key is missing in config.
        ValueError: If the path is empty or invalid.
    """
    logger.debug("Extracting embedding parameters")

    # Check if config is a dictionary
    if not isinstance(config, dict):
        logger.error("Config parameter for embedding_store must be a dictionary")
        raise TypeError("Config must be a dictionary")

    # Check if the required key exists in config
    if "generated_embedding_store" not in config:
        logger.error("'generated_embedding_store' key missing in config")
        raise KeyError("'generated_embedding_store' key is required")

    # Get the path from config (could be string or Path object)
    path_value = config.get("generated_embedding_store")
    
    # Check if the path value is None or empty
    if not path_value:
        logger.error("Path value is empty or None")
        raise ValueError("Path value cannot be empty or None")

    try:
        # Convert to Path object if it's a string, or use as-is if already a Path
        if isinstance(path_value, str):
            embedding_store = Path(path_value)
        elif isinstance(path_value, Path):
            embedding_store = path_value
        else:
            logger.error(f"Invalid path type: expected string or Path, got {type(path_value).__name__}")
            raise TypeError(f"Invalid path type: expected string or Path, got {type(path_value).__name__}")
        
        # Resolve the path to handle relative paths and ensure it's absolute
        embedding_store = embedding_store.resolve()
        
        # Check if the path is a directory (optional, depending on your use case)
        if embedding_store.exists() and not embedding_store.is_dir():
            logger.warning(f"Path {embedding_store} exists but is not a directory")
        
        # Create the directory if it doesn't exist (optional, based on your needs)
        embedding_store.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Embedding store path set to: {embedding_store}")
        
        return embedding_store
    
    except (OSError, ValueError) as e:
        logger.error(f"Invalid path '{path_value}': {str(e)}")
        raise ValueError(f"Invalid path '{path_value}': {str(e)}")


def get_domain_mapped_csv_store(config: Dict) -> Optional[Path]:
    """Extract Domain Mapped CSV generation parameters and return a Path object.

    Args:
        config: A dictionary containing configuration parameters.

    Returns:
        Path: A Path object representing the Domain Mapped CSV store directory, or None if invalid.

    Raises:
        KeyError: If 'domain_mapped_csv_store' key is missing in config.
        ValueError: If the path is empty or invalid.
    """
    logger.debug("Extracting Domain Mapped CSV parameters")

    # Check if config is a dictionary
    if not isinstance(config, dict):
        logger.error("Config parameter domain_mapped_csv_store must be a dictionary")
        logger.error(type(config))
        raise TypeError("Config must be a dictionary")

    # Check if the required key exists in config
    if "domain_mapped_csv_store" not in config:
        logger.error("'domain_mapped_csv_store' key missing in config")
        raise KeyError("'domain_mapped_csv_store' key is required")

    # Get the path from config (could be string or Path object)
    path_value = config.get("domain_mapped_csv_store")
    
    # Check if the path value is None or empty
    if not path_value:
        logger.error("Path value is empty or None")
        raise ValueError("Path value cannot be empty or None")

    try:
        # Convert to Path object if it's a string, or use as-is if already a Path
        if isinstance(path_value, str):
            domain_mapped_csv = Path(path_value)
        elif isinstance(path_value, Path):
            domain_mapped_csv = path_value
        else:
            logger.error(f"Invalid path type: expected string or Path, got {type(path_value).__name__}")
            raise TypeError(f"Invalid path type: expected string or Path, got {type(path_value).__name__}")
        
        # Resolve the path to handle relative paths and ensure it's absolute
        domain_mapped_csv = domain_mapped_csv.resolve()
        
        # Check if the path is a directory (optional, depending on your use case)
        if domain_mapped_csv.exists() and not domain_mapped_csv.is_dir():
            logger.warning(f"Path {domain_mapped_csv} exists but is not a directory")
        
        # Create the directory if it doesn't exist (optional, based on your needs)
        domain_mapped_csv.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Domain Mapped CSV store path set to: {domain_mapped_csv}")
        
        return domain_mapped_csv
    
    except (OSError, ValueError) as e:
        logger.error(f"Invalid path '{path_value}': {str(e)}")
        raise ValueError(f"Invalid path '{path_value}': {str(e)}")


def _coerce_store_path(path_value, label: str) -> Path:
    """Internal helper: turn string/Path into absolute Path, mkdir -p."""
    if not path_value:
        raise ValueError(f"{label} path cannot be empty or None")
    if isinstance(path_value, str):
        p = Path(path_value)
    elif isinstance(path_value, Path):
        p = path_value
    else:
        raise TypeError(f"{label} must be a string or Path, got {type(path_value).__name__}")
    p = p.resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def get_company_mapped_data_processed_data_store(config: dict) -> Path:
    """Return company_mapped_data.processed_data_store as a Path (mkdir -p)."""
    cmd = config.get("company_mapped_data")
    if not isinstance(cmd, dict):
        raise KeyError("'company_mapped_data' section missing or not a dict in config")
    return _coerce_store_path(cmd.get("processed_data_store"),
                              "company_mapped_data.processed_data_store")

def get_company_mapped_data_tfidf_search_store(config: dict) -> Path:
    """Return company_mapped_data.tfidf_search_store as a Path (mkdir -p)."""
    cmd = config.get("company_mapped_data")
    if not isinstance(cmd, dict):
        raise KeyError("'company_mapped_data' section missing or not a dict in config")
    return _coerce_store_path(cmd.get("tfidf_search_store"),
                              "company_mapped_data.tfidf_search_store")

def get_company_mapped_data_dense_index_store(config: dict) -> Path:
    """Return company_mapped_data.dense_index_store as a Path (mkdir -p)."""
    cmd = config.get("company_mapped_data")
    if not isinstance(cmd, dict):
        raise KeyError("'company_mapped_data' section missing or not a dict in config")
    return _coerce_store_path(cmd.get("dense_index_store"),
                              "company_mapped_data.dense_index_store")

    
def get_search_params(config):
    """Extract search-related parameters"""
    logger.debug("Extracting search parameters")
    return {
        'build_embeddings': config.get('build_embeddings', config.get('domain_mapped_csv_store')),
        'query': config['query'],
        'embeddings': config['embeddings'],
        'top_k': config['top_k'],
        'stores_root': config.get('generated_embedding_store'),
        'text_cols': config['text_cols'],
        'min_score': config['min_score'],
        'exact_only': config['exact_only'],
        'prefer_exact': config['prefer_exact'],
        'boost_exact': config['boost_exact'],
        'columns': config['columns'],
        'show_source': config['show_source']
    } if config else None

if __name__ == "__main__":
    try:
        # Example usage - now returns both config and logger
        (config, logger) = load_config()
        
        # Get SQL connection parameters
        sql_params = get_sql_connection_params(config)
        logger.info(f"SQL Connection Parameters: {sql_params}")
        
        # Get SQL processing parameters
        sql_processing_params = get_sql_processing_params(config)
        logger.info(f"SQL Processing Parameters: {sql_processing_params}")
        
        # Get processing parameters
        processing_params = get_processing_params(config)
        logger.info(f"Processing Parameters: {processing_params}")
        
        # Get embedding parameters
        embedding_params = get_embedding_params(config)
        logger.info(f"Embedding Parameters: {embedding_params}")
        
        # Get search parameters
        search_params = get_search_params(config)
        logger.info(f"Search Parameters: {search_params}")
        
        # Get the domain mapped CSV store
        domain_mapped_csv_store = get_domain_mapped_csv_store(config)
        if not domain_mapped_csv_store:
            logger.error("Failed to get the domain mapped CSV store path")

        # Get the company mapped store
        company_mapped_store = get_company_mapped_data_processed_data_store(config)
        if not company_mapped_store:
            logger.error("Failed to get the company mapped store path")

        # Get the company mapped store
        company_mapped_tfidf_store = get_company_mapped_data_tfidf_search_store(config)
        if not company_mapped_tfidf_store:
            logger.error("Failed to get the company mapped tfidf store path")

        # Get the company mapped store
        company_mapped_dense_index_store = get_company_mapped_data_dense_index_store(config)
        if not company_mapped_dense_index_store:
            logger.error("Failed to get the company mapped dense index store path")

        # Construct MSSQL connection string
        if sql_params:
            mssql_conn = get_mssql_connection_string(sql_params)
            logger.info(f"MSSQL Connection String: {mssql_conn}")
        else:
            logger.info("No SQL connection parameters available")
    except (FileNotFoundError, NotADirectoryError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Error: {str(e)}")
        raise
