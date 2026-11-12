"""Offline checks for source acquisition recovery. No network/model execution."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch,Mock
import zipfile
import pandas as pd
import requests
import urllib.error
import ssl
from io import BytesIO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import design_predictors as dp

class DownloadRecoveryChecks(unittest.TestCase):
    def setUp(self):
        # Unit tests do not need the OS trust-store contents; live acquisition
        # verifies the real factory. This avoids sandbox-specific trust-store I/O.
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        patched=patch.object(dp.ssl,'create_default_context',return_value=context)
        patched.start();self.addCleanup(patched.stop)

    def workbook(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('[Content_Types].xml','<Types/>')
            z.writestr('xl/workbook.xml','<workbook/>')

    def test_pipeline_downloads_without_browser_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp); original=home/'fixture/laucnty18.xlsx'
            self.workbook(original)
            target=home/'project/data/raw/laus/laucnty18.xlsx'
            response=BytesIO(original.read_bytes())
            response.headers={'Content-Length':str(original.stat().st_size)}
            response.geturl=lambda:dp.SOURCES['laus']['url']
            with patch.object(dp.urllib.request,'urlopen',return_value=response) as get,patch.object(dp,'_browser_download',side_effect=AssertionError('Manual browser fallback must not be used')):
                self.assertEqual(dp.fetch(dp.SOURCES['laus']['url'],target),target)
                request=get.call_args.args[0]
                self.assertIn('RegionalLaborResilience',request.get_header('User-agent'))
                self.assertTrue(get.call_args.kwargs['context'].check_hostname)
                self.assertEqual(get.call_args.kwargs['context'].verify_mode,ssl.CERT_REQUIRED)
            self.assertEqual(original.read_bytes(),target.read_bytes())
            self.assertTrue(target.with_suffix('.xlsx.download.json').exists())

    def test_forbidden_retains_reason_without_demanding_manual_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp); target=home/'project/laucnty18.xlsx'
            error=urllib.error.HTTPError(dp.SOURCES['laus']['url'],403,'Forbidden',{},None)
            with patch.object(dp.urllib.request,'urlopen',side_effect=error) as get:
                with self.assertRaisesRegex(RuntimeError,'HTTP 403') as exc:dp.fetch(dp.SOURCES['laus']['url'],target)
                self.assertNotIn('normal browser',str(exc.exception));self.assertEqual(get.call_count,1)
            self.assertFalse(target.exists());self.assertFalse(target.with_suffix('.xlsx.part').exists())

    def test_html_cache_is_not_accepted_as_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'laucnty18.xlsx';target.write_text('<html>Access denied</html>')
            with patch.object(dp.urllib.request,'urlopen') as get:
                with self.assertRaisesRegex(ValueError,'Invalid cached source'):dp.fetch(dp.SOURCES['laus']['url'],target)
                get.assert_not_called()

    def test_tls_failure_does_not_disable_certificate_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            home=Path(tmp);target=home/'cache/laucnty18.xlsx'
            with patch.object(dp.urllib.request,'urlopen',side_effect=ssl.SSLCertVerificationError('bad issuer')) as get:
                with self.assertRaisesRegex(RuntimeError,'TLS certificate verification failed'):dp.fetch(dp.SOURCES['laus']['url'],target)
                self.assertTrue(get.call_args.kwargs['context'].check_hostname)

    def test_published_oes_layout_rejects_partial_county_townships(self):
        table=pd.DataFrame([
            ['FIPS code','State','State abbreviation','May 2018 MSA code','May 2018 MSA name','County code','Township code'],
            ['46','South Dakota','SD','10100','County-defined area','013','000'],
            ['09','Connecticut','CT','71650','Town-defined area','001','12345']])
        with patch.object(dp.pd,'read_excel',return_value=table):
            got=dp.oes_county_sets(Path('fixture.xlsx'))
        self.assertEqual(got.loc['10100'],{'46013'})
        self.assertNotIn('71650',got.index)

if __name__=='__main__':unittest.main()
