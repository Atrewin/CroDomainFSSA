#! /bin/bash
## author:Haopeng Ren
## This is an shell file to extract the structured data

## Extract the reviews from the xml files

# -----------------------------------TODO MASK Task Data Format --------------------------------------

# TODO book-electronics
source_unlabeled_xml_url=data/domain_data/init_data/electronics/unlabeled.review    # Total number of training steps
target_unlabeled_xml_url=data/domain_data/init_data/books/book.unlabeled    # Warmup the learning rate over this many updates
sentence_pair_keep_url=extension/Post-bert/data/domain_data/processed_data/book2electronics

python3 data_extractor.py --source_unlabeled_xml_url $source_unlabeled_xml_url --target_unlabeled_xml_url $target_unlabeled_xml_url \
--sentence_pair_keep_url $sentence_pair_keep_url --process_mode MASK


# TODO book-dvd
source_unlabeled_xml_url=data/domain_data/init_data/dvd/unlabeled.review    # Total number of training steps
target_unlabeled_xml_url=data/domain_data/init_data/books/book.unlabeled    # Warmup the learning rate over this many updates
sentence_pair_keep_url=extension/Post-bert/data/domain_data/processed_data/book2dvd

python3 data_extractor.py --source_unlabeled_xml_url $source_unlabeled_xml_url --target_unlabeled_xml_url $target_unlabeled_xml_url \
--sentence_pair_keep_url $sentence_pair_keep_url --process_mode MASK


# TODO book-kitchen
source_unlabeled_xml_url=data/domain_data/init_data/kitchenAndhousewares/unlabeled.review    # Total number of training steps
target_unlabeled_xml_url=data/domain_data/init_data/books/book.unlabeled    # Warmup the learning rate over this many updates
sentence_pair_keep_url=extension/Post-bert/data/domain_data/processed_data/book2kitchen

python3 data_extractor.py --source_unlabeled_xml_url $source_unlabeled_xml_url --target_unlabeled_xml_url $target_unlabeled_xml_url \
--sentence_pair_keep_url $sentence_pair_keep_url --process_mode MASK


# TODO dvd-electronics
source_unlabeled_xml_url=data/domain_data/init_data/electronics/unlabeled.review    # Total number of training steps
target_unlabeled_xml_url=data/domain_data/init_data/dvd/unlabeled.review    # Warmup the learning rate over this many updates
sentence_pair_keep_url=extension/Post-bert/data/domain_data/processed_data/dvd2electronics

python3 data_extractor.py --source_unlabeled_xml_url $source_unlabeled_xml_url --target_unlabeled_xml_url $target_unlabeled_xml_url \
--sentence_pair_keep_url $sentence_pair_keep_url --process_mode MASK


# TODO dvd-kitchen
source_unlabeled_xml_url=data/domain_data/init_data/kitchenAndhousewares/unlabeled.review    # Total number of training steps
target_unlabeled_xml_url=data/domain_data/init_data/dvd/unlabeled.review    # Warmup the learning rate over this many updates
sentence_pair_keep_url=extension/Post-bert/data/domain_data/processed_data/dvd2kitchen

python3 data_extractor.py --source_unlabeled_xml_url $source_unlabeled_xml_url --target_unlabeled_xml_url $target_unlabeled_xml_url \
--sentence_pair_keep_url $sentence_pair_keep_url --process_mode MASK


# TODO electronics-kitchen
source_unlabeled_xml_url=data/domain_data/init_data/kitchenAndhousewares/unlabeled.review    # Total number of training steps
target_unlabeled_xml_url=data/domain_data/init_data/electronics/unlabeled.review    # Warmup the learning rate over this many updates
sentence_pair_keep_url=extension/Post-bert/data/domain_data/processed_data/electronics2kitchen

python3 data_extractor.py --source_unlabeled_xml_url $source_unlabeled_xml_url --target_unlabeled_xml_url $target_unlabeled_xml_url \
--sentence_pair_keep_url $sentence_pair_keep_url --process_mode MASK

