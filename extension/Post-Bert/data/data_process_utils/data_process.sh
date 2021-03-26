#! /bin/bash
## author:Haopeng Ren
## This is an shell file to extract the structured data

## Extract the reviews from the xml files
python3 data_extractor.py --source_unlabeled_xml_url ../domain_data/init_data/kitchenAndhousewares/unlabeled.review --target_unlabeled_xml_url ../domain_data/init_data/books/book.unlabeled \
--sentence_pair_keep_url ../domain_data/processed_data/book2kitchen --process_mode MASK

python3 data_extractor.py --source_unlabeled_xml_url ../domain_data/init_data/kitchenAndhousewares/unlabeled.review --target_unlabeled_xml_url ../domain_data/init_data/books/book.unlabeled \
--sentence_pair_keep_url ../domain_data/processed_data/book2kitchen --process_mode DSP