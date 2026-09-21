"""Match every cloud history object to regular files and aliases in local tar files."""
from pathlib import Path, PurePosixPath
import hashlib, json, posixpath, tarfile
from huggingface_hub import HfApi

BASE = Path(__file__).resolve().parent
revision = json.loads((BASE/'recovery.json').read_text())['revision']
info = HfApi().model_info('a1390892757/junqi-guarded-continue-20260914', revision=revision, files_metadata=True)
assert info.private
lookup = {x.rfilename: x for x in info.siblings}
out = {}
for arm in ('N', 'M', 'R'):
    hashes, links = {}, {}
    with tarfile.open(BASE/'full_history'/f'{arm}.tar.gz', mode='r|gz') as tar:
        for member in tar:
            if member.isfile():
                sha256 = hashlib.sha256()
                blob = hashlib.sha1(b'blob '+str(member.size).encode()+b'\0')
                with tar.extractfile(member) as handle:
                    while chunk := handle.read(4*1024*1024):
                        sha256.update(chunk)
                        blob.update(chunk)
                hashes[member.name] = (member.size, sha256.hexdigest(), blob.hexdigest())
            elif member.islnk():
                links[member.name] = member.linkname
            elif member.issym():
                links[member.name] = posixpath.normpath(str(PurePosixPath(member.name).parent/member.linkname))
    while links:
        ready = [name for name, target in links.items() if target in hashes]
        assert ready, links
        for name in ready:
            hashes[name] = hashes[links.pop(name)]
    prefix = f'feature_focus_20260921/{arm}/complete/'
    expected = {n: x for n, x in lookup.items() if n.startswith(prefix)}
    for name, item in expected.items():
        local = arm+'/'+name[len(prefix):]
        size, sha256, blob = hashes[local]
        assert size == item.size, name
        assert sha256 == item.lfs.sha256 if item.lfs else blob == item.blob_id, name
    out[arm] = {'cloud_files_verified': len(expected), 'hashes_match_local_archive': True}
report = {'private': info.private, 'revision': revision, 'arms': out}
(BASE/'hf_full_history_verified.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
