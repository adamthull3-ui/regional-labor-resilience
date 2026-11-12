"""Offline checks for source acquisition recovery. No network/model execution."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
import zipfile
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import design_predictors as dp

class DownloadRecoveryChecks(unittest.TestCase):
    def workbook(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('[Content_Types].xml','<Types/>')
            z.writestr('xl/workbook.xml','<workbook/>')

    def test_browser_download_adopted_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp); original=home/'Downloads/laucnty18.xlsx'
            self.workbook(original)
            target=home/'project/data/raw/laus/laucnty18.xlsx'
            with patch.object(dp.Path,'home',return_value=home),patch.dict(dp.os.environ,{'RL_SOURCE_DOWNLOAD_DIR':''}),patch.object(dp.requests,'get') as get:
                self.assertEqual(dp.fetch(dp.SOURCES['laus']['url'],target),target)
                get.assert_not_called()
            self.assertEqual(original.read_bytes(),target.read_bytes())

    def test_forbidden_retains_http_reason_and_manual_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp); target=home/'project/laucnty18.xlsx'
            response=Mock(); response.status_code=403
            response.raise_for_status.side_effect=requests.HTTPError(response=response)
            response.__enter__=Mock(return_value=response);response.__exit__=Mock(return_value=False)
            with patch.object(dp.Path,'home',return_value=home),patch.dict(dp.os.environ,{'RL_SOURCE_DOWNLOAD_DIR':''}),patch.object(dp.requests,'get',return_value=response) as get:
                with self.assertRaisesRegex(RuntimeError,'HTTP 403') as exc:dp.fetch(dp.SOURCES['laus']['url'],target)
                self.assertIn('Downloads',str(exc.exception));self.assertEqual(get.call_count,1)
            self.assertFalse(target.exists());self.assertFalse(target.with_suffix('.xlsx.part').exists())

    def test_html_cache_is_not_accepted_as_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'laucnty18.xlsx';target.write_text('<html>Access denied</html>')
            with patch.object(dp.requests,'get') as get:
                with self.assertRaisesRegex(ValueError,'Invalid cached source'):dp.fetch(dp.SOURCES['laus']['url'],target)
                get.assert_not_called()

    def test_tls_failure_does_not_disable_certificate_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);target=home/'cache/laucnty18.xlsx'
            with patch.object(dp.Path,'home',return_value=home),patch.dict(dp.os.environ,{'RL_SOURCE_DOWNLOAD_DIR':''}),patch.object(dp.requests,'get',side_effect=requests.exceptions.SSLError('bad issuer')) as get:
                with self.assertRaisesRegex(RuntimeError,'TLS certificate verification failed'):dp.fetch(dp.SOURCES['laus']['url'],target)
                self.assertNotIn('verify',get.call_args.kwargs)

if __name__=='__main__':unittest.main()
