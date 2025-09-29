import requests
f='c:\\Windows\\Temp\\tiny.pdf'
print('Posting', f)
r=requests.post('http://127.0.0.1:5000/api/analyze', files={'file':open(f,'rb')})
print('Status', r.status_code)
print('Text', r.text[:2000])
