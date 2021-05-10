#! /bin/bash
## author:Haopeng Ren
## This is an shell file to extract the structured data

## Extract the reviews from the xml files
# Can execute script from anywhere
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../..
#
#CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 data/data_process_utils/data_extractor.py --pos_xml_url data/domain_data/init_data/books/positive.review --neg_xml_url data/domain_data/init_data/books/negative.review --keep_url data/domain_data/processed_data/books/book_reviews.json \
#--unlabeled_xml_url data/domain_data/init_data/books/book.unlabeled --unlabeled_keep_url data/domain_data/processed_data/books/book_unlabeled_reviews.json

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 data/data_process_utils/data_extractor.py --pos_xml_url data/domain_data/init_data/dvd/positive.review --neg_xml_url data/domain_data/init_data/dvd/negative.review --keep_url data/domain_data/processed_data/dvd/dvd_reviews.json \
--unlabeled_xml_url data/domain_data/init_data/dvd/unlabeled.review --unlabeled_keep_url data/domain_data/processed_data/dvd/dvd_unlabeled_reviews.json

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 data/data_process_utils/data_extractor.py --pos_xml_url data/domain_data/init_data/electronics/positive.review --neg_xml_url data/domain_data/init_data/electronics/negative.review --keep_url data/domain_data/processed_data/electronics/electronics_reviews.json \
--unlabeled_xml_url data/domain_data/init_data/electronics/unlabeled.review --unlabeled_keep_url data/domain_data/processed_data/electronics/electronics_unlabeled_reviews.json
#
#CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python3 data/data_process_utils/data_extractor.py --pos_xml_url data/domain_data/init_data/kitchenAndhousewares/positive.review --neg_xml_url data/domain_data/init_data/kitchenAndhousewares/negative.review --keep_url data/domain_data/processed_data/kitchenAndhousewares/kitchen_reviews.json \
#--unlabeled_xml_url data/domain_data/init_data/kitchenAndhousewares/unlabeled.review --unlabeled_keep_url data/domain_data/processed_data/kitchenAndhousewares/kitchen_unlabeled_reviews.json