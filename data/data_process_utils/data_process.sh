#! /bin/bash
## author:Haopeng Ren
## This is an shell file to extract the structured data

## Extract the reviews from the xml files
python3 data_extractor.py --pos_xml_url data/domain_data/init_data/books/positive.review --neg_xml_url data/domain_data/init_data/books/negative.review --keep_url data/book_reviews.json \
--unlabeled_xml_url data/domain_data/init_data/books/book.unlabeled --unlabeled_keep_url data/book_unlabeled_reviews.json
python3 data_extractor.py --pos_xml_url data/domain_data/init_data/dvd/positive.review --neg_xml_url data/domain_data/init_data/dvd/negative.review --keep_url data/dvd_reviews.json \
--unlabeled_xml_url data/domain_data/init_data/dvd/unlabeled.review --unlabeled_keep_url data/dvd_unlabeled_reviews.json
python3 data_extractor.py --pos_xml_url data/domain_data/init_data/electronics/positive.review --neg_xml_url data/domain_data/init_data/electronics/negative.review --keep_url data/electronics_reviews.json \
--unlabeled_xml_url data/domain_data/init_data/electronics/unlabeled.review --unlabeled_keep_url data/electronics_unlabeled_reviews.json
python3 data_extractor.py --pos_xml_url data/domain_data/init_data/kitchenAndhousewares/positive.review --neg_xml_url data/domain_data/init_data/kitchenAndhousewares/negative.review --keep_url data/kitchen_reviews.json \
--unlabeled_xml_url data/domain_data/init_data/kitchenAndhousewares/unlabeled.review --unlabeled_keep_url data/kitchen_unlabeled_reviews.json