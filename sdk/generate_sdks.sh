python convert_schemas.py

echo "Python"
quicktype -o python/recoai_visual_search/models.py --src-lang schema --python-version 3.7 schemas/*.json
