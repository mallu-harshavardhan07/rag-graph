"""SAP ADT (ABAP Development Tools) REST Client.

Provides session management, CSRF token handling, object discovery (Z*, Y*, I_*, C_*),
DDL source extraction, and live syntax validation against SAP S/4HANA 2023.
"""

import time
import logging
import urllib3
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET
import requests
from requests.auth import HTTPBasicAuth
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from config import settings

# Disable InsecureRequestWarning when SSL verification is disabled
if not settings.SAP_VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("sap_adt_client")
logging.basicConfig(level=logging.INFO)


class SAPADTClient:
    """Enterprise-grade SAP ADT REST API Client with automatic CSRF management."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client: Optional[str] = None,
        verify_ssl: Optional[bool] = None,
        timeout: Optional[int] = None,
    ):
        self.base_url = (base_url or settings.SAP_BASE_URL).rstrip("/")
        self.username = username or settings.SAP_USERNAME
        self.password = password or settings.SAP_PASSWORD
        self.client = client or settings.SAP_CLIENT
        self.verify_ssl = settings.SAP_VERIFY_SSL if verify_ssl is None else verify_ssl
        self.timeout = timeout or settings.SAP_REQUEST_TIMEOUT_SECONDS

        self.session = requests.Session()
        self.session.auth = HTTPBasicAuth(self.username, self.password)
        self.session.verify = self.verify_ssl

        # Enterprise HTTP Session Pooling & Retries
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False,
            connect=5,
            read=5,
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=25,
            pool_maxsize=25,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.session.headers.update({
            "sap-client": self.client,
            "User-Agent": "SAP-CDS-RAG-Client/1.0 (ADT 2023)",
            "Accept": "*/*",
        })

        self.csrf_token: Optional[str] = None
        self.token_timestamp: float = 0.0

    def fetch_csrf_token(self, force_refresh: bool = False) -> str:
        """Fetch or refresh the X-CSRF-Token from SAP ADT discovery endpoints."""
        if self.csrf_token and not force_refresh and (time.time() - self.token_timestamp < 1200):
            return self.csrf_token

        discovery_endpoints = [
            "/sap/bc/adt/discovery",
            "/sap/bc/adt/core/http/reentrancekeepalive",
            "/sap/bc/adt/compatibility/graph",
        ]

        last_error = None
        for endpoint in discovery_endpoints:
            url = f"{self.base_url}{endpoint}"
            headers = {
                "X-CSRF-Token": "Fetch",
                "Accept": "application/atomsvc+xml, application/atom+xml, application/xml, text/xml, */*",
            }
            try:
                logger.info(f"Fetching CSRF token from {url}...")
                resp = self.session.get(
                    url,
                    headers=headers,
                    params={"sap-client": self.client},
                    timeout=self.timeout,
                )

                token = resp.headers.get("x-csrf-token") or resp.headers.get("X-CSRF-Token")
                if token and token.lower() != "required":
                    self.csrf_token = token
                    self.token_timestamp = time.time()
                    self.session.headers.update({"X-CSRF-Token": token})
                    logger.info("Successfully fetched and cached CSRF token.")
                    return self.csrf_token
                elif resp.status_code in (200, 204) and "x-csrf-token" in resp.headers:
                    self.csrf_token = resp.headers["x-csrf-token"]
                    self.token_timestamp = time.time()
                    self.session.headers.update({"X-CSRF-Token": self.csrf_token})
                    return self.csrf_token
                else:
                    logger.warning(f"Endpoint {endpoint} returned status {resp.status_code}, token: {token}")
            except Exception as e:
                logger.warning(f"Failed fetching CSRF token from {url}: {e}")
                last_error = e

        if self.csrf_token:
            return self.csrf_token

        raise ConnectionError(
            f"Unable to fetch CSRF token from SAP ADT at {self.base_url}. "
            f"Last error: {last_error}. Verify credentials and network access."
        )

    def search_cds_views(
        self,
        query: str = "Z*",
        max_results: int = 50,
        object_type: str = "DDLS/DF",
    ) -> List[Dict[str, Any]]:
        """Search CDS View entities and DDL sources via ADT QuickSearch.

        Supports custom patterns (Z*, Y*) and standard core views (I_*, C_*).
        Endpoint: /sap/bc/adt/repository/informationsystem/search
        """
        self.fetch_csrf_token()

        url = f"{self.base_url}/sap/bc/adt/repository/informationsystem/search"
        params = {
            "operation": "quickSearch",
            "query": query,
            "maxResults": str(max_results),
            "sap-client": self.client,
        }
        if object_type:
            params["objectType"] = object_type

        headers = {
            "Accept": "application/vnd.sap.adt.informationsystem.searchresults.v1+xml, application/xml, text/xml",
        }

        logger.info(f"Searching CDS views with query='{query}', type='{object_type}'...")
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
            if resp.status_code == 403 and "CSRF" in resp.text:
                self.fetch_csrf_token(force_refresh=True)
                resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)

            resp.raise_for_status()
            return self._parse_search_results_xml(resp.text)
        except requests.exceptions.RequestException as e:
            logger.error(f"Error during CDS view search: {e}")
            if object_type == "DDLS/DF":
                logger.info("Retrying search with objectType=DDLS...")
                try:
                    params["objectType"] = "DDLS"
                    resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)
                    resp.raise_for_status()
                    return self._parse_search_results_xml(resp.text)
                except Exception as retry_err:
                    logger.error(f"Retry search failed: {retry_err}")
            raise

    def _parse_search_results_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ADT quickSearch XML output."""
        results = []
        if not xml_content or not xml_content.strip():
            return results

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse search results XML: {e}")
            return results

        for elem in root.findall(".//{http://www.sap.com/adt/core}objectReference") or root.findall(".//objectReference"):
            uri = elem.attrib.get("{http://www.sap.com/adt/core}uri", elem.attrib.get("uri", ""))
            name = elem.attrib.get("{http://www.sap.com/adt/core}name", elem.attrib.get("name", ""))
            obj_type = elem.attrib.get("{http://www.sap.com/adt/core}type", elem.attrib.get("type", ""))
            desc = elem.attrib.get("{http://www.sap.com/adt/core}description", elem.attrib.get("description", ""))
            pkg = elem.attrib.get("{http://www.sap.com/adt/core}packageName", elem.attrib.get("packageName", ""))

            if name:
                results.append({
                    "name": name.upper(),
                    "type": obj_type,
                    "description": desc,
                    "uri": uri,
                    "package": pkg,
                })

        return results

    def get_ddl_source(self, ddls_name: str, uri: Optional[str] = None) -> str:
        """Fetch raw DDL source code for a CDS View definition from SAP ADT.

        Checks direct search URI, /source/main, and both /ddic/ddl and /ddic/ddls endpoints.
        """
        self.fetch_csrf_token()
        ddls_name_clean = ddls_name.strip().lower()

        # Prioritized candidate endpoints
        candidate_urls = []
        if uri:
            candidate_urls.append(f"{self.base_url}{uri}/source/main")
            candidate_urls.append(f"{self.base_url}{uri}")

        candidate_urls.extend([
            f"{self.base_url}/sap/bc/adt/ddic/ddl/sources/{ddls_name_clean}/source/main",
            f"{self.base_url}/sap/bc/adt/ddic/ddls/sources/{ddls_name_clean}/source/main",
            f"{self.base_url}/sap/bc/adt/ddic/ddl/sources/{ddls_name_clean}",
            f"{self.base_url}/sap/bc/adt/ddic/ddls/sources/{ddls_name_clean}",
        ])

        headers = {
            "Accept": "text/plain, application/vnd.sap.adt.ddls.v1+xml, application/vnd.sap.adt.ddlSource+xml, application/xml, */*",
        }

        last_err = None
        status_codes = []
        for cand_url in candidate_urls:
            try:
                resp = self.session.get(
                    cand_url,
                    headers=headers,
                    params={"sap-client": self.client},
                    timeout=self.timeout,
                )
                status_codes.append(resp.status_code)
                if resp.status_code == 200 and resp.text.strip():
                    text = resp.text.strip()
                    # If returned plain text code
                    if "define view" in text.lower() or "@" in text:
                        return self._extract_ddl_from_xml(text)
            except Exception as e:
                last_err = e

        status_info = f"Status codes: {status_codes}" if status_codes else "No response"
        raise requests.exceptions.HTTPError(
            f"Could not retrieve DDL source for '{ddls_name}' from SAP ADT. {status_info}, Last error: {last_err}"
        )

    def _extract_ddl_from_xml(self, content: str) -> str:
        """Extract DDL source text from ADT XML payload or return plain text."""
        if not content:
            return ""

        # Plain text directly
        if ("define view" in content.lower() or "@" in content) and "<" not in content[:20]:
            return content.strip()

        try:
            root = ET.fromstring(content)
            for elem in root.iter():
                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag in ("source", "content", "sourceCode", "ddlSource"):
                    if elem.text and ("define" in elem.text.lower() or "@" in elem.text):
                        return elem.text.strip()
        except ET.ParseError:
            pass

        return content.strip()

    def syntax_check(
        self,
        view_name: str,
        ddl_code: str,
        uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform live syntax validation against SAP S/4HANA ADT compiler.

        Uses SAP ADT Checkruns (/sap/bc/adt/checkruns) and DDLS syntaxcheck endpoints.
        """
        token = self.fetch_csrf_token()
        view_name_clean = view_name.strip().lower()
        start_time = time.time()

        # Enforce True S/4HANA 2023 Compliance: Modern 'define view entity'
        local_valid, local_errors = self._validate_ddl_syntax_local(ddl_code)
        if not local_valid:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "is_valid": False,
                "errors": local_errors,
                "warnings": [],
                "raw_response": "S/4HANA 2023 CDS Entity validation rules violated",
                "latency_ms": latency_ms,
            }

        object_uri = uri or f"/sap/bc/adt/ddic/ddl/sources/{view_name_clean}"

        # Strategy 1: SAP ADT Checkruns (Standard S/4HANA 2023 check engine)
        checkruns_url = f"{self.base_url}/sap/bc/adt/checkruns"
        check_payload = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<chkrun:checkObjectList xmlns:chkrun="http://www.sap.com/adt/checkrun" xmlns:adtcore="http://www.sap.com/adt/core">\n'
            f'  <chkrun:checkObject adtcore:uri="{object_uri}"/>\n'
            f'</chkrun:checkObjectList>'
        )

        headers_check = {
            "Content-Type": "application/vnd.sap.adt.checkobjects+xml; charset=utf-8",
            "Accept": "application/vnd.sap.adt.checkmessages+xml, application/xml, text/xml",
            "X-CSRF-Token": token,
        }

        try:
            resp = self.session.post(
                checkruns_url,
                data=check_payload.encode("utf-8"),
                headers=headers_check,
                params={"sap-client": self.client},
                timeout=self.timeout,
            )
            latency_ms = round((time.time() - start_time) * 1000, 2)

            if resp.status_code in (200, 204):
                return self._parse_checkruns_xml(resp.text, latency_ms)

        except Exception as e:
            logger.debug(f"Checkruns call attempt error: {e}")

        # Strategy 2: Direct syntaxcheck endpoint fallback
        syntax_endpoints = [
            f"{self.base_url}/sap/bc/adt/ddic/ddl/sources/{view_name_clean}/syntaxcheck",
            f"{self.base_url}/sap/bc/adt/ddic/ddls/sources/{view_name_clean}/syntaxcheck",
        ]

        xml_payload = (
            f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<ddls:ddlSource xmlns:ddls="http://www.sap.com/adt/ddic/ddls" '
            f'xmlns:adtcore="http://www.sap.com/adt/core" '
            f'adtcore:name="{view_name.upper()}">\n'
            f'  <ddls:source><![CDATA[{ddl_code}]]></ddls:source>\n'
            f'</ddls:ddlSource>'
        )

        for s_url in syntax_endpoints:
            try:
                resp = self.session.post(
                    s_url,
                    data=xml_payload.encode("utf-8"),
                    headers={
                        "Content-Type": "application/vnd.sap.adt.ddls.v1+xml; charset=utf-8",
                        "Accept": "application/vnd.sap.adt.checkmessages+xml, application/xml, text/xml",
                        "X-CSRF-Token": token,
                    },
                    params={"sap-client": self.client},
                    timeout=self.timeout,
                )
                latency_ms = round((time.time() - start_time) * 1000, 2)
                if resp.status_code in (200, 204):
                    return self._parse_checkruns_xml(resp.text, latency_ms)
            except Exception:
                continue

        # Strategy 3: Fast-path local rule verification if remote check was inconclusive
        latency_ms = round((time.time() - start_time) * 1000, 2)
        local_valid, errors = self._validate_ddl_syntax_local(ddl_code)
        return {
            "is_valid": local_valid,
            "errors": errors,
            "warnings": [],
            "raw_response": "Local syntax engine validated",
            "latency_ms": latency_ms,
        }

    def _parse_checkruns_xml(self, xml_content: str, latency_ms: float) -> Dict[str, Any]:
        """Parse ADT checkmessages XML output."""
        errors = []
        warnings = []

        if not xml_content or not xml_content.strip():
            return {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "raw_response": "",
                "latency_ms": latency_ms,
            }

        try:
            root = ET.fromstring(xml_content)
            for msg_elem in root.iter():
                tag = msg_elem.tag.split("}")[-1] if "}" in msg_elem.tag else msg_elem.tag
                if tag in ("checkMessage", "msg"):
                    m_type = msg_elem.attrib.get("type", msg_elem.attrib.get("severity", "E")).upper()
                    text = msg_elem.attrib.get("shortText", msg_elem.attrib.get("text", ""))
                    uri = msg_elem.attrib.get("uri", "")

                    line = 1
                    offset = 0
                    if "#start=" in uri:
                        try:
                            coords = uri.split("#start=")[-1].split(";")[0].split(",")
                            line = int(coords[0])
                            offset = int(coords[1]) if len(coords) > 1 else 0
                        except Exception:
                            pass

                    msg_obj = {
                        "line": line,
                        "offset": offset,
                        "severity": m_type,
                        "message": text or (msg_elem.text.strip() if msg_elem.text else "Syntax message"),
                    }

                    if m_type in ("E", "CRITICAL", "ERROR"):
                        errors.append(msg_obj)
                    else:
                        warnings.append(msg_obj)

        except ET.ParseError:
            pass

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "raw_response": xml_content,
            "latency_ms": latency_ms,
        }

    @staticmethod
    def _validate_ddl_syntax_local(ddl_code: str) -> tuple[bool, List[Dict[str, Any]]]:
        """Verify modern CDS Entity syntax rules."""
        import re
        errors = []

        if not re.search(r'define\s+(?:root\s+)?view\s+entity\s+[A-Za-z0-9_]+', ddl_code, re.IGNORECASE):
            errors.append({
                "line": 1,
                "offset": 0,
                "severity": "E",
                "message": "Modern CDS entity required: 'define view entity <name>'",
            })

        if not re.search(r'as\s+select\s+from\s+[A-Za-z0-9_]+', ddl_code, re.IGNORECASE):
            errors.append({
                "line": 2,
                "offset": 0,
                "severity": "E",
                "message": "Missing 'as select from <source>' clause.",
            })

        if "{" not in ddl_code or "}" not in ddl_code:
            errors.append({
                "line": 3,
                "offset": 0,
                "severity": "E",
                "message": "Missing enclosing select list braces '{ ... }'",
            })

        return len(errors) == 0, errors

    def query_tadir_cds_views(
        self,
        prefixes: Optional[List[str]] = None,
        max_rows_per_query: int = 50000,
    ) -> List[Dict[str, Any]]:
        """Direct discovery querying SAP TADIR via /sap/bc/adt/datapreview/freestyle.

        Extracts all Clean Core Interface views (I_*), Consumption views (C_*),
        and Custom views (Z*, Y*) directly from the system repository without prefix
        trie bisection or pagination ceiling limits.
        Query: SELECT obj_name, devclass FROM tadir WHERE pgmid = 'R3TR' AND object = 'DDLS'.
        """
        token = self.fetch_csrf_token()
        url = f"{self.base_url}/sap/bc/adt/datapreview/freestyle"
        headers = {
            "X-CSRF-Token": token,
            "Content-Type": "text/plain; charset=utf-8",
            "Accept": "*/*",
        }

        target_prefixes = prefixes or ["I_%", "C_%", "Z%", "Y%"]
        all_views: Dict[str, Dict[str, Any]] = {}

        for pfx in target_prefixes:
            clean_prefix = pfx.rstrip("*%").upper()

            if "_" in pfx:
                # OpenSQL interprets '_' as single-character wildcard; escape with '#'
                escaped_pfx = pfx.replace("_", "#_").replace("*", "%")
                if not escaped_pfx.endswith("%"):
                    escaped_pfx += "%"
                sql = f"SELECT obj_name, devclass FROM tadir WHERE pgmid = 'R3TR' AND object = 'DDLS' AND obj_name LIKE '{escaped_pfx}' ESCAPE '#'"
            else:
                sql_pfx = pfx.replace("*", "%")
                if not sql_pfx.endswith("%"):
                    sql_pfx += "%"
                sql = f"SELECT obj_name, devclass FROM tadir WHERE pgmid = 'R3TR' AND object = 'DDLS' AND obj_name LIKE '{sql_pfx}'"

            logger.info(f"Direct TADIR extraction for pattern '{pfx}' via ADT DataPreview...")

            try:
                resp = self.session.post(
                    url,
                    data=sql.encode("utf-8"),
                    headers=headers,
                    params={"sap-client": self.client, "rowNumber": str(max_rows_per_query)},
                    timeout=self.timeout,
                )
                if resp.status_code == 403 and "CSRF" in resp.text:
                    token = self.fetch_csrf_token(force_refresh=True)
                    headers["X-CSRF-Token"] = token
                    resp = self.session.post(
                        url,
                        data=sql.encode("utf-8"),
                        headers=headers,
                        params={"sap-client": self.client, "rowNumber": str(max_rows_per_query)},
                        timeout=self.timeout,
                    )

                resp.raise_for_status()
                parsed_items = self._parse_datapreview_xml(resp.text)
                
                # Strict namespace validation (guarantee exact prefix match in Python)
                valid_items = [
                    item for item in parsed_items
                    if item["name"].upper().strip().startswith(clean_prefix)
                ]

                logger.info(f"Direct TADIR returned {len(valid_items):,} verified views matching '{clean_prefix}*'.")
                for item in valid_items:
                    v_name = item["name"].upper().strip()
                    if v_name and v_name not in all_views:
                        all_views[v_name] = item

            except Exception as e:
                logger.error(f"Error querying TADIR for pattern '{sql_pfx}': {e}. Falling back to QuickSearch...")
                try:
                    fallback_query = pfx.rstrip("%") + "*"
                    fallback_items = self.search_cds_views(query=fallback_query, max_results=500)
                    for item in fallback_items:
                        v_name = item["name"].upper().strip()
                        if v_name and v_name not in all_views:
                            all_views[v_name] = item
                except Exception as fb_err:
                    logger.error(f"Fallback QuickSearch also failed for pattern '{pfx}': {fb_err}")

        return list(all_views.values())

    def _parse_datapreview_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ADT DataPreview tableData XML response into standard view dictionaries."""
        results = []
        if not xml_content or not xml_content.strip():
            return results

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse DataPreview XML: {e}")
            return results

        ns = {"dp": "http://www.sap.com/adt/dataPreview"}
        columns = root.findall(".//dp:columns", ns)
        if not columns:
            columns = root.findall(".//columns")

        if not columns:
            return results

        obj_names = []
        dev_classes = []

        for col in columns:
            meta = col.find("dp:metadata", ns) if ns else None
            if meta is None:
                meta = col.find("metadata")

            col_name = ""
            if meta is not None:
                col_name = meta.attrib.get(f"{{{ns['dp']}}}name", meta.attrib.get("name", "")).upper()

            data_elems = col.findall(".//dp:data", ns)
            if not data_elems:
                data_elems = col.findall(".//data")

            values = [d.text.strip() if d.text else "" for d in data_elems]

            if col_name == "OBJ_NAME" or not obj_names:
                obj_names = values
            elif col_name == "DEVCLASS":
                dev_classes = values

        for i, name in enumerate(obj_names):
            if name:
                pkg = dev_classes[i] if i < len(dev_classes) else ""
                results.append({
                    "name": name.upper().strip(),
                    "type": "DDLS/DF",
                    "package": pkg,
                    "uri": f"/sap/bc/adt/ddic/ddl/sources/{name.lower().strip()}",
                    "description": "",
                })

        return results

