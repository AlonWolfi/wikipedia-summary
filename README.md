# wikipedia-summary

### Get Data ###
1. go to page https://en.wikipedia.org/wiki/Special:Export
2. enter 'List of multi-sport athletes' in "Add pages manually"
3. save the XML file as 'data_list.xml' in the /data folder
4. run "data_list_extractor.py'
5. enter the list again into "Add pages manually" in https://en.wikipedia.org/wiki/Special:Export
6. save the list as 'raw_data.xml' in the /data folder
7. run "data_extractor.py"

### Model ###
1. run "run_pipeline.py"