# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from flask import Flask
from flask import abort, jsonify, request
from flask_cors import CORS

import boto3
import json
import os
import pprint

from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.flask.middleware import XRayMiddleware
from aws_xray_sdk.core import patch_all

patch_all()

from typing import Dict, List, Tuple, Union
from flask import Flask, jsonify, Response
from flask import request

from flask_cors import CORS
from experimentation.experiment_manager import ExperimentManager
from experimentation.resolvers import DefaultProductResolver, PersonalizeRecommendationsResolver, \
    PersonalizeRankingResolver, RankingProductsNoOpResolver, PersonalizeContextComparePickResolver, RandomPickResolver
from experimentation.utils import CompatEncoder
from expiring_dict import ExpiringDict

import json
import os
import pprint
import boto3
import requests
import random
import logging
from datetime import datetime

from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.ext.flask.middleware import XRayMiddleware
from aws_xray_sdk.core import patch_all

# patch_all()

xray_recorder.begin_segment("Videos-init")

import logging
import json
import os
import pathlib
import pprint
import subprocess
import threading
import time

import boto3
import srt
from flask import Flask, jsonify, Response
from flask_cors import CORS



RESOURCE_BUCKET = os.environ.get('RESOURCE_BUCKET')

s3 = boto3.resource('s3')

store_location = {}
customer_route = {}

cstore_location = {}
cstore_route = {}


def load_s3_data():
    global customer_route
    route_file_obj = s3.Object(RESOURCE_BUCKET, 'location_services/customer_route.json')
    customer_route = json.loads(route_file_obj.get()['Body'].read().decode('utf-8'))

    global store_location
    location_file_obj = s3.Object(RESOURCE_BUCKET, 'location_services/store_location.json')
    store_location = json.loads(location_file_obj.get()['Body'].read().decode('utf-8'))

    global cstore_route
    route_file_obj = s3.Object(RESOURCE_BUCKET, 'location_services/cstore_route.json')
    cstore_route = json.loads(route_file_obj.get()['Body'].read().decode('utf-8'))

    global cstore_location
    route_file_obj = s3.Object(RESOURCE_BUCKET, 'location_services/cstore_location.json')
    cstore_location = json.loads(route_file_obj.get()['Body'].read().decode('utf-8'))


# -- Logging
class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)
# -- End Logging


# -- Handlers
app = Flask(__name__)
corps = CORS(app)


@app.route('/')
def index():
    return 'Location Service Service'


@app.route('/store_location')
def get_store_location():
    return jsonify(store_location)


@app.route('/customer_route')
def get_customer_route():
    return jsonify(customer_route)


@app.route('/cstore_location')
def get_cstore_location():
    return jsonify(cstore_location)


@app.route('/cstore_route')
def get_cstore_route():
    return jsonify(cstore_route)


if __name__ == '__main__':
    app.wsgi_app = LoggingMiddleware(app.wsgi_app)
    load_s3_data()

    app.run(debug=True, host='0.0.0.0', port=80)

offers = []


def load_offers():
    global offers
    with open('data/offers.json') as f:
        offers = json.load(f)


# -- Logging
class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)
# -- End Logging


# -- Handlers
app = Flask(__name__)
corps = CORS(app)


@app.route('/')
def index():
    return 'Offers Service'


@app.route('/offers')
def get_offers():
    return jsonify({'tasks': offers})


@app.route('/offers/<offer_id>')
def get_offer(offer_id):
    for offer in offers:
        if offer['id'] == int(offer_id):
            return jsonify({'task': offer})
    abort(404)


if __name__ == '__main__':
    app.wsgi_app = LoggingMiddleware(app.wsgi_app)

    load_offers()
    app.run(debug=True, host='0.0.0.0', port=80)

NUM_DISCOUNTS = 2

EXPERIMENTATION_LOGGING = True
DEBUG_LOGGING = True

random.seed(42)  # Keep our demonstration deterministic

# Since the DescribeRecommender/DescribeCampaign APIs easily throttles and we just
# need the recipe from the recommender/campaign and it won't change often (if at all),
# use a cache to help smooth out periods where we get throttled.
personalize_meta_cache = ExpiringDict(2 * 60 * 60)

servicediscovery = boto3.client('servicediscovery')
personalize = boto3.client('personalize')
personalize_runtime = boto3.client('personalize-runtime')
ssm = boto3.client('ssm')
codepipeline = boto3.client('codepipeline')
sts = boto3.client('sts')
cw_events = boto3.client('events')

# SSM parameter name for the Personalize filter for purchased and c-store items
filter_purchased_param_name = '/retaildemostore/personalize/filters/filter-purchased-arn'
filter_cstore_param_name = '/retaildemostore/personalize/filters/filter-cstore-arn'
filter_purchased_cstore_param_name = '/retaildemostore/personalize/filters/filter-purchased-and-cstore-arn'
filter_include_categories_param_name = '/retaildemostore/personalize/filters/filter-include-categories-arn'
promotion_filter_param_name = '/retaildemostore/personalize/filters/promoted-items-filter-arn'
promotion_filter_no_cstore_param_name = '/retaildemostore/personalize/filters/promoted-items-no-cstore-filter-arn'
offers_arn_param_name = '/retaildemostore/personalize/personalized-offers-arn'

# -- Shared Functions

def get_recipe(arn):
    """ Returns the Amazon Personalize recipe ARN for the specified campaign/recommender ARN """
    recipe = None

    is_recommender = arn.split(':')[5].startswith('recommender/')

    resource = personalize_meta_cache.get(arn)
    if not resource:
        if is_recommender:
            response = personalize.describe_recommender(recommenderArn = arn)
            if response.get('recommender'):
                resource = response['recommender']
                personalize_meta_cache[arn] = resource
        else:
            response = personalize.describe_campaign(campaignArn = arn)
            if response.get('campaign'):
                resource = response['campaign']
                personalize_meta_cache[arn] = resource

    if resource:
        if is_recommender:
            recipe = resource['recipeArn']
        else:
            solution_version = personalize_meta_cache.get(resource['solutionVersionArn'])

            if not solution_version:
                response = personalize.describe_solution_version(solutionVersionArn = resource['solutionVersionArn'])
                if response.get('solutionVersion'):
                    solution_version = response['solutionVersion']
                    personalize_meta_cache[resource['solutionVersionArn']] = solution_version

            if solution_version:
                recipe = solution_version['recipeArn']

    return recipe

def get_parameter_values(names):
    """ Returns values for SSM parameters or None for params that don't exist or that have value equal 'NONE' """
    if isinstance(names, str):
        names = [ names ]

    response = ssm.get_parameters(Names = names)

    values = []

    for name in names:
        found = False
        for param in response['Parameters']:
            if param['Name'] == name:
                found = True
                if param['Value'] != 'NONE':
                    values.append(param['Value'])
                else:
                    values.append(None)
                break

        if not found:
            values.append(None)

    assert len(values) == len(names), 'mismatch in number of values returned for names'

    return values

def get_timestamp_from_request() -> datetime:
    timestamp_raw = request.args.get('timestamp')
    if not timestamp_raw and request.method == 'POST':
        if request.is_json:
            timestamp_raw = request.json.get('timestamp')
        elif request.content_type.startswith('application/x-www-form-urlencoded'):
            timestamp_raw = request.form.get('timestamp')

    timestamp: datetime = None
    if timestamp_raw:
        if isinstance(timestamp_raw, str) and not timestamp_raw.isnumeric():
            raise BadRequest('timestamp is not numeric (must be unix time)')
        timestamp = datetime.fromtimestamp(int(timestamp_raw))

    return timestamp

def get_products_service_host_and_port() -> Tuple[str, int]:
    """ Returns a tuple of the products service host name and port """
    # Check environment for host and port first in case we're running in a local Docker container (dev mode)
    products_service_host = os.environ.get('PRODUCT_SERVICE_HOST')
    products_service_port = os.environ.get('PRODUCT_SERVICE_PORT', 80)

    if not products_service_host:
        # Get product service instance. We'll need it rehydrate product info for recommendations.
        response = servicediscovery.discover_instances(
            NamespaceName='retaildemostore.local',
            ServiceName='products',
            MaxResults=1,
            HealthStatus='HEALTHY'
        )

        products_service_host = response['Instances'][0]['Attributes']['AWS_INSTANCE_IPV4']

    return products_service_host, products_service_port

def fetch_product_details(item_ids: Union[str, List[str]], fully_qualify_image_urls=False) -> List[Dict]:
    """ Fetches details for one or more products from the products service """
    products_service_host, products_service_port = get_products_service_host_and_port()

    item_ids_csv = item_ids if isinstance(item_ids, str) else ','.join(item_ids)

    url = f'http://{products_service_host}:{products_service_port}/products/id/{item_ids_csv}?fullyQualifyImageUrls={fully_qualify_image_urls}'
    app.logger.debug(f"Asking for product info from {url}")

    products = []

    response = requests.get(url)
    if response.ok:
        products = response.json()
        if not isinstance(products, list):
            products = [ products ]

    return products

def get_products(feature, user_id, current_item_id, num_results, default_inference_arn_param_name,
                 default_filter_arn_param_name, filter_values=None, user_reqd_for_inference=False, fully_qualify_image_urls=False,
                 promotion: Dict = None
                 ):
    """ Returns products given a UI feature, user, item/product.

    If a feature name is provided and there is an active experiment for the
    feature, the experiment will be used to retrieve products. Otherwise,
    the default behavior will be used which will look to see if an Amazon Personalize
    campaign/recommender is available. If not, the Product service will be called to get products
    from the same category as the current product.
    Args:
        feature: Used to track different experiments - different experiments pertain to different features
        user_id: If supplied we are looking at user personalization
        current_item_id: Or maybe we are looking at related items
        num_results: Num to return
        default_inference_arn_param_name: If no experiment active, use this SSM parameters to get recommender Arn
        default_filter_arn_param_name: If no experiment active, use this SSM parameter to get filter Arn, if exists
        filter_values: Values to pass at inference for the filter
        user_reqd_for_inference: Require a user ID to use Personalze - otherwise default
        fully_qualify_image_urls: Fully qualify image URLs n here
        promotion: Personalize promotional filter configuration
    Returns:
        A prepared HTTP response object.
    """

    items = []
    resp_headers = {}
    experiment = None
    exp_manager = None

    # Get active experiment if one is setup for feature and we have a user.
    if feature and user_id:
        exp_manager = ExperimentManager()
        experiment = exp_manager.get_active(feature, user_id)

    if experiment:
        # Get items from experiment.
        tracker = exp_manager.default_tracker()

        items = experiment.get_items(
            user_id = user_id,
            current_item_id = current_item_id,
            num_results = num_results,
            tracker = tracker,
            filter_values = filter_values,
            timestamp = get_timestamp_from_request(),
            promotion = promotion
        )

        resp_headers['X-Experiment-Name'] = experiment.name
        resp_headers['X-Experiment-Type'] = experiment.type
        resp_headers['X-Experiment-Id'] = experiment.id
    else:
        # Fallback to default behavior of checking for campaign/recommender ARN parameter and
        # then the default product resolver.
        values = get_parameter_values([default_inference_arn_param_name, default_filter_arn_param_name])

        inference_arn = values[0]
        filter_arn = values[1]

        if inference_arn and (user_id or not user_reqd_for_inference):

            logger.info(f"get_products: Supplied campaign/recommender: {inference_arn} (from {default_inference_arn_param_name}) Supplied filter: {filter_arn} (from {default_filter_arn_param_name}) Supplied user: {user_id}")

            resolver = PersonalizeRecommendationsResolver(inference_arn = inference_arn, filter_arn = filter_arn)

            items = resolver.get_items(
                user_id = user_id,
                product_id = current_item_id,
                num_results = num_results,
                filter_values = filter_values,
                promotion = promotion
            )

            resp_headers['X-Personalize-Recipe'] = get_recipe(inference_arn)
        else:
            products_service_host, products_service_port = get_products_service_host_and_port()
            resolver = DefaultProductResolver(products_service_host = products_service_host, products_service_port = products_service_port)

            items = resolver.get_items(product_id = current_item_id, num_results = num_results)

    item_ids = [item['itemId'] for item in items]

    products = fetch_product_details(item_ids, fully_qualify_image_urls)
    for item in items:
        item_id = item['itemId']

        product = next((p for p in products if p['id'] == item_id), None)
        if product is not None and 'experiment' in item and 'url' in product:
            # Append the experiment correlation ID to the product URL so it gets tracked if used by client.
            product_url = product.get('url')
            if '?' in product_url:
                product_url += '&'
            else:
                product_url += '?'

            product_url += 'exp=' + item['experiment']['correlationId']

            product['url'] = product_url

        item.update({
            'product': product
        })

        item.pop('itemId')

    return items, resp_headers

# -- Logging
class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)

# -- End Logging

# -- Exceptions
class BadRequest(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

# -- Handlers

app = Flask(__name__)
logger = app.logger
corps = CORS(app, expose_headers=['X-Experiment-Name', 'X-Experiment-Type', 'X-Experiment-Id', 'X-Personalize-Recipe'])

xray_recorder.configure(service='Recommendations Service')
XRayMiddleware(app, xray_recorder)

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/')
def index():
    return 'Recommendations Service'

@app.route('/health')
def health():
    return 'OK'

@app.route('/related', methods=['GET'])
def related():
    """ Returns related products given an item/product.

    If a feature name is provided and there is an active experiment for the
    feature, the experiment will be used to retrieve related products. Otherwise,
    the default behavior will be used which will look to see if an Amazon Personalize
    campaign/recommender for related items is available. If not, the Product service
    will be called to get products from the same category as the current product.
    """
    user_id = request.args.get('userID')

    current_item_id = request.args.get('currentItemID')
    if not current_item_id:
        raise BadRequest('currentItemID is required')

    num_results = request.args.get('numResults', default = 25, type = int)
    if num_results < 1:
        raise BadRequest('numResults must be greater than zero')
    if num_results > 100:
        raise BadRequest('numResults must be less than 100')

    # The default filter includes products from the same category as the current item.
    filter_ssm = request.args.get('filter', filter_include_categories_param_name)
    # We have short names for these filters
    if filter_ssm == 'cstore': filter_ssm = filter_cstore_param_name
    elif filter_ssm == 'purchased': filter_ssm = filter_purchased_param_name
    app.logger.info("Filter SSM for /related: %s", filter_ssm)

    filter_values = None
    if filter_ssm == filter_include_categories_param_name:
        category = request.args.get('currentItemCategory')
        if not category:
            products = fetch_product_details(current_item_id)
            if products:
                category = products[0]['category']

        filter_values = { "CATEGORIES": f"\"{category}\"" }

    # Determine name of feature where related items are being displayed
    feature = request.args.get('feature')

    fully_qualify_image_urls = request.args.get('fullyQualifyImageUrls', '0').lower() in [ 'true', 't', '1']

    try:
        # If a user ID is provided, automatically perform reranking of related items to personalize items (composite use case).
        if user_id:
            rerank_items = True
            related_count = num_results * 3
        else:
            rerank_items = False
            related_count = num_results

        items, resp_headers = get_products(
            feature = feature,
            user_id = user_id,
            current_item_id = current_item_id,
            num_results = related_count,
            default_inference_arn_param_name='/retaildemostore/personalize/related-items-arn',
            default_filter_arn_param_name=filter_ssm,
            filter_values=filter_values,
            fully_qualify_image_urls = fully_qualify_image_urls
        )

        if rerank_items:
            app.logger.info('Reranking related items to personalize order for user %s', user_id)
            items, resp_headers = get_ranking(user_id, items, feature = None, resp_headers = resp_headers)

            items = items[0:num_results]    # Trim back down to the requested number of items.

        resp = Response(json.dumps(items, cls=CompatEncoder), content_type = 'application/json', headers = resp_headers)
        return resp

    except Exception as e:
        app.logger.exception('Unexpected error generating related items', e)
        raise BadRequest(message = 'Unhandled error', status_code = 500)

@app.route('/recommendations', methods=['GET'])
def recommendations():
    """ Returns item/product recommendations for a given user in the context
    of a current item (e.g. the user is viewing a product and I want to provide
    recommendations for other products they may be interested in).

    If an experiment is currently active for this feature ('home_product_recs'),
    recommendations will be provided by the experiment. Otherwise, the default
    behavior will be used which will look to see if an Amazon Personalize
    campaign/recommender is available. If not, the Product service will be called to get
    products from the same category as the current product or featured products.
    """
    user_id = request.args.get('userID')
    if not user_id:
        raise BadRequest('userID is required')

    current_item_id = request.args.get('currentItemID')

    num_results = request.args.get('numResults', default = 25, type = int)
    if num_results < 1:
        raise BadRequest('numResults must be greater than zero')
    if num_results > 100:
        raise BadRequest('numResults must be less than 100')

    # Determine name of feature where related items are being displayed
    feature = request.args.get('feature')

    # The default filter is the not-already-purchased filter
    filter_ssm = request.args.get('filter', filter_purchased_param_name)
    # We have short names for these filters
    if filter_ssm == 'cstore': filter_ssm = filter_cstore_param_name
    elif filter_ssm == 'purchased': filter_ssm = filter_purchased_param_name
    app.logger.info(f"Filter SSM for /recommendations: {filter_ssm}")

    fully_qualify_image_urls = request.args.get('fullyQualifyImageUrls', '0').lower() in [ 'true', 't', '1']

    promotion = None
    promotion_filter_arn = get_parameter_values(promotion_filter_param_name)[0]
    if promotion_filter_arn:
        promotion = {
            'name': 'promotedItem',
            'percentPromotedItems': 25,
            'filterArn': promotion_filter_arn
        }

    try:
        items, resp_headers = get_products(
            feature = feature,
            user_id = user_id,
            current_item_id = current_item_id,
            num_results = num_results,
            default_inference_arn_param_name='/retaildemostore/personalize/recommended-for-you-arn',
            default_filter_arn_param_name=filter_ssm,
            fully_qualify_image_urls = fully_qualify_image_urls,
            promotion = promotion
        )

        response = Response(json.dumps(items, cls=CompatEncoder), content_type = 'application/json', headers = resp_headers)

        app.logger.debug("Recommendations response to be returned: %s", response)
        return response

    except Exception as e:
        app.logger.exception('Unexpected error generating recommendations', e)
        raise BadRequest(message = 'Unhandled error', status_code = 500)

@app.route('/popular', methods=['GET'])
def popular():
    """ Returns item/product recommendations for a given user in the context
    of a current item (e.g. the user is viewing a product and I want to provide
    recommendations for other products they may be interested in).

    If an experiment is currently active for this feature ('home_product_recs'),
    recommendations will be provided by the experiment. Otherwise, the default
    behavior will be used which will look to see if an Amazon Personalize
    campaign/recommender is available. If not, the Product service will be called to get
    products from the same category as the current product or featured products.
    """
    user_id = request.args.get('userID')
    if not user_id:
        raise BadRequest('userID is required')

    current_item_id = request.args.get('currentItemID')

    num_results = request.args.get('numResults', default = 25, type = int)
    if num_results < 1:
        raise BadRequest('numResults must be greater than zero')
    if num_results > 100:
        raise BadRequest('numResults must be less than 100')

    # Determine name of feature where related items are being displayed
    feature = request.args.get('feature')

    # The default filter is the exclude already purchased and c-store products filter
    filter_ssm = request.args.get('filter', filter_purchased_cstore_param_name)
    # We have short names for these filters
    if filter_ssm == 'cstore': filter_ssm = filter_cstore_param_name
    elif filter_ssm == 'purchased': filter_ssm = filter_purchased_cstore_param_name
    app.logger.info(f"Filter SSM for /recommendations: {filter_ssm}")

    fully_qualify_image_urls = request.args.get('fullyQualifyImageUrls', '0').lower() in [ 'true', 't', '1']

    promotion = None
    promotion_filter_arn = get_parameter_values(promotion_filter_no_cstore_param_name)[0]
    if promotion_filter_arn:
        promotion = {
            'name': 'promotedItem',
            'percentPromotedItems': 25,
            'filterArn': promotion_filter_arn
        }

    try:
        items, resp_headers = get_products(
            feature = feature,
            user_id = user_id,
            current_item_id = current_item_id,
            num_results = num_results,
            default_inference_arn_param_name='/retaildemostore/personalize/popular-items-arn',
            default_filter_arn_param_name=filter_ssm,
            fully_qualify_image_urls = fully_qualify_image_urls,
            promotion = promotion
        )

        response = Response(json.dumps(items, cls=CompatEncoder), content_type = 'application/json', headers = resp_headers)

        app.logger.debug("Recommendations response to be returned: %s", response)
        return response

    except Exception as e:
        app.logger.exception('Unexpected error generating recommendations', e)
        raise BadRequest(message = 'Unhandled error', status_code = 500)

def ranking_request_params():
    """
    Utility function which grabs a JSON body and extracts the UserID, item list and feature name.
    Returns:
        3-tuple of user ID, item list and feature name
    """

    content = request.json
    app.logger.info(f"JSON payload: {content}")

    user_id = content.get('userID')
    if not user_id:
        raise BadRequest('userID is required')

    items = content.get('items')
    if not items:
        raise BadRequest('items is required')

    # Determine name of feature where reranked items are being displayed
    feature = content.get('feature')
    if not feature:
        feature = request.args.get('feature')

    app.logger.info(f"Items pulled from json: {items}")

    return user_id, items, feature

def get_ranking(user_id, items, feature,
                default_inference_arn_param_name='/retaildemostore/personalize/personalized-ranking-arn',
                top_n=None, context=None, resp_headers=None):
    """
    Re-ranks a list of items using personalized reranking.
    Or delegates to experiment manager if there is an active experiment.

    Args:
        user_id (int):
        items (list[dict]): e.g. [{"itemId":"33", "url":"path_to_product33"},
                                  {"itemId":"22", "url":"path_to_product22"}]
        feature: Used to lookup the currently active experiment.
        default_inference_arn_param_name: For discounts this would be different.
        top_n (Optional[int]): Only return the top N ranked if not None.
        context (Optional[dict]): If available, passed to the reranking Personalization recipe.
        resp_headers (Optional[dict]): Response headers from chained call

    Returns:
        Items as passed in, but ordered according to reranker - also might have experimentation metadata added.
    """

    app.logger.info(f"Items given for ranking: {items}")

    # Extract item IDs from items supplied by caller. Note that unranked items
    # can be specified as a list of objects with just an 'itemId' key or as a
    # list of fully defined items/products (i.e. with an 'id' key).
    item_map = {}
    unranked_items = []
    for item in items:
        item_id = item.get('itemId')
        if not item_id:
            item_id = item.get('id')
        if not item_id and item.get('product'):
            item_id = item.get('product').get('id')
        item_map[item_id] = item
        unranked_items.append(item_id)

    app.logger.info(f"Unranked items: {unranked_items}")

    if resp_headers is None:
        resp_headers = {}

    experiment = None
    exp_manager = None

    # Get active experiment if one is setup for feature.
    if feature:
        exp_manager = ExperimentManager()
        experiment = exp_manager.get_active(feature, user_id)

    if experiment:
        app.logger.info('Using experiment: %s', experiment.name)

        # Get ranked items from experiment.
        tracker = exp_manager.default_tracker()

        ranked_items = experiment.get_items(
            user_id=user_id,
            item_list=unranked_items,
            tracker=tracker,
            context=context,
            timestamp=get_timestamp_from_request()
        )

        app.logger.debug("Experiment ranking resolver gave us this ranking: %s", ranked_items)

        resp_headers['X-Experiment-Name'] = experiment.name
        resp_headers['X-Experiment-Type'] = experiment.type
        resp_headers['X-Experiment-Id'] = experiment.id
    else:
        # Fallback to default behavior of checking for campaign/recommender ARN parameter and
        # then the default product resolver.
        values = get_parameter_values([default_inference_arn_param_name, filter_purchased_param_name])
        app.logger.info(f'Falling back to Personalize: {values}')

        inference_arn = values[0]
        filter_arn = values[1]

        if inference_arn:
            resolver = PersonalizeRankingResolver(inference_arn=inference_arn, filter_arn=filter_arn)
            recipe_arn = get_recipe(inference_arn)
            if resp_headers.get('X-Personalize-Recipe'):
                resp_headers['X-Personalize-Recipe'] = resp_headers['X-Personalize-Recipe'] + ',' + recipe_arn
            else:
                resp_headers['X-Personalize-Recipe'] = recipe_arn
        else:
            app.logger.info(f'Falling back to No-op: {values}')
            resolver = RankingProductsNoOpResolver()

        ranked_items = resolver.get_items(
            user_id=user_id,
            product_list=unranked_items,
            context=context
        )

    response_items = []
    if top_n is not None:
        # We may not want to return them all - for example in a "pick the top N" scenario.
        ranked_items = ranked_items[:top_n]

    for ranked_item in ranked_items:
        # Unlike with /recommendations and /related we are not hitting the products API to get product info back
        # The caller may have left that info in there so in case they have we want to leave it in.
        item = item_map.get(ranked_item.get('itemId'))

        if 'experiment' in ranked_item:

            item['experiment'] = ranked_item['experiment']

            if 'url' in item:
                # Append the experiment correlation ID to the product URL so it gets tracked if used by client.
                product_url = item.get('url')
                if '?' in product_url:
                    product_url += '&'
                else:
                    product_url += '?'

                product_url += 'exp=' + ranked_item['experiment']['correlationId']

                item['url'] = product_url

        response_items.append(item)

    return response_items, resp_headers

@app.route('/rerank', methods=['POST'])
def rerank():
    """
    Gets user ID, items list and feature and gets ranking of items according to reranking campaign.
    """
    items = []
    try:
        user_id, items, feature = ranking_request_params()
        print('ITEMS', items)
        response_items, resp_headers = get_ranking(user_id, items, feature)
        app.logger.debug(f"Response items for reranking: {response_items}")
        resp = Response(json.dumps(response_items, cls=CompatEncoder), content_type='application/json',
                        headers=resp_headers)
        return resp
    except Exception as e:
        app.logger.exception('Unexpected error reranking items', e)
        return json.dumps(items)


def get_top_n(user_id, items, feature, top_n,
            default_inference_arn_param_name='/retaildemostore/personalize/personalized-ranking-arn'):
    """
    Gets Top N items using provided campaign/recommender.
    Or delegates to experiment manager if there is an active experiment.

    Args:
        user_id (int): User to get the topN for
        items (list[dict]): e.g. [{"itemId":"33", "url":"path_to_product33"},
                                  {"itemId":"22", "url":"path_to_product22"}]
        feature: Used to lookup the currently active experiment.
        top_n (int): Only return the top N ranked if not None.
        default_inference_arn_param_name: Change this to use a different campaign/recommender.

    Returns:
        Items as passed in, but truncated according to picker - also might have experimentation metadata added.
    """

    app.logger.info(f"Items given for top-n: {items}")

    # Extract item IDs from items supplied by caller. Note that unranked items
    # can be specified as a list of objects with just an 'itemId' key or as a
    # list of fully defined items/products (i.e. with an 'id' key).
    item_map = {}
    unranked_items = []
    for item in items:
        item_id = item.get('itemId') if item.get('itemId') else item.get('id')
        item_map[item_id] = item
        unranked_items.append(item_id)

    app.logger.info(f"Pre-selection items: {unranked_items}")

    resp_headers = {}
    experiment = None
    exp_manager = None

    # Get active experiment if one is setup for feature.
    if feature:
        exp_manager = ExperimentManager()
        experiment = exp_manager.get_active(feature, user_id)

    if experiment:
        app.logger.info('Using experiment: ' + experiment.name)

        # Get ranked items from experiment.
        tracker = exp_manager.default_tracker()

        topn_items = experiment.get_items(
            user_id=user_id,
            item_list=unranked_items,
            tracker=tracker,
            num_results=top_n,
            timestamp=get_timestamp_from_request()
        )

        app.logger.debug(f"Experiment ranking resolver gave us this ranking: {topn_items}")

        resp_headers['X-Experiment-Name'] = experiment.name
        resp_headers['X-Experiment-Type'] = experiment.type
        resp_headers['X-Experiment-Id'] = experiment.id
    else:
        # Fallback to default behavior of checking for campaign/recommender ARN parameter and
        # then the default product resolver.
        values = get_parameter_values([default_inference_arn_param_name, filter_purchased_param_name])
        app.logger.info(f'Falling back to Personalize: {values}')

        inference_arn = values[0]
        filter_arn = values[1]

        if inference_arn:
            resolver = PersonalizeContextComparePickResolver(inference_arn=inference_arn, filter_arn=filter_arn,
                                                             with_context={'Discount': 'Yes'},
                                                             without_context={})
            resp_headers['X-Personalize-Recipe'] = get_recipe(inference_arn)
        else:
            app.logger.info(f'Falling back to No-op: {values}')
            resolver = RandomPickResolver()

        topn_items = resolver.get_items(
            user_id=user_id,
            product_list=unranked_items,
            num_results=top_n
        )

    logger.info(f"Sorted items: returned from resolver: {topn_items}")

    response_items = []

    for top_item in topn_items:
        # Unlike with /recommendations and /related we are not hitting the products API to get product info back
        # The caller may have left that info in there so in case they have we want to leave it in.
        item_id = top_item['itemId']
        item = item_map[item_id]

        if 'experiment' in top_item:

            item['experiment'] = top_item['experiment']

            if 'url' in item:
                # Append the experiment correlation ID to the product URL so it gets tracked if used by client.
                product_url = item.get('url')
                if '?' in product_url:
                    product_url += '&'
                else:
                    product_url += '?'

                product_url += 'exp=' + top_item['experiment']['correlationId']

                item['url'] = product_url

        response_items.append(item)

    logger.info(f"Top-N response: with details added back in: {topn_items}")

    return response_items, resp_headers


@app.route('/choose_discounted', methods=['POST'])
def choose_discounted():
    """
    Gets user ID, items list and feature and chooses which items to discount according to the
    reranking campaign. Gets a ranking with discount applied and without (using contextual metadata)
    and looks at the difference. The products are ordered according to how the response is expected
    to improve after applying discount.

    The items that are not chosen for discount will be returned as-is but with the "discounted" key set to False.
    The items that are chosen for discount will have the "discounted" key set to True.

    If there is an experiment active for this feature the request for ranking for choosing discounts will have been
    routed through the experiment resolver and discounts chosen according to whichever approach is active. The
    items will have experiment information recorded against them and if URLs were provided for products these will be
    suffixed with an experiment tracking correlation ID. That way, different approaches to discounting can be compared,
    as with different approaches to recommendations and reranking in other campaigns.
    """
    items = []
    try:
        user_id, items, feature = ranking_request_params()
        response_items, resp_headers = get_top_n(user_id, items, feature, NUM_DISCOUNTS)
        discount_item_map = {item['itemId']: item for item in response_items}

        return_items = []
        for item in items:
            item_id = item['itemId']
            if item_id in discount_item_map:
                # This was picked for discount so we flag it as a discounted item. It may also have experiment
                # information recorded against it by get_ranking() if an experiment is active.
                discounted_item = discount_item_map[item_id]
                discounted_item['discounted'] = True
                return_items.append(discounted_item)
            else:
                # This was not picked for discount, so is not participating in any experiment comparing
                # discount approaches and we also do not flag it as a discounted item
                item['discounted'] = False
                return_items.append(item)

        resp = Response(json.dumps(items, cls=CompatEncoder), content_type='application/json',
                        headers=resp_headers)
        return resp
    except Exception as e:
        app.logger.exception('Unexpected error calculating discounted items', e)
        return json.dumps(items)


def get_offers_service():
    """
    Get offers service URL root. Check for env variables first in case we're running in a local Docker container (dev mode)
    """

    service_host = os.environ.get('OFFERS_SERVICE_HOST')
    service_port = os.environ.get('OFFERS_SERVICE_PORT', 80)

    if not service_host or service_host.strip().lower() == 'offers.retaildemostore.local':
        # Get product service instance. We'll need it rehydrate product info for recommendations.
        response = servicediscovery.discover_instances(
            NamespaceName='retaildemostore.local',
            ServiceName='offers',
            MaxResults=1,
            HealthStatus='HEALTHY'
        )

        service_host = response['Instances'][0]['Attributes']['AWS_INSTANCE_IPV4']

    return service_host, service_port


def get_all_offers_by_id():
    """We might wish to prepopulate all offers if we are going to be picking up multiple offers."""
    offers_service_host, offers_service_port = get_offers_service()
    url = f'http://{offers_service_host}:{offers_service_port}/offers'
    logger.debug(f"Asking for offers info from {url}")
    offers_response = requests.get(url)  # we let connection error propagate
    logger.debug(f"Got offer info: {offers_response}")
    if not offers_response.ok:
        logger.error(f"Offers service not giving us offers: {offers_response.reason}")
        raise BadRequest(message='Cannot obtain offers', status_code=500)
    offers = offers_response.json()['tasks']
    offers_by_id = {str(offer['id']): offer for offer in offers}
    return offers_by_id


def get_offer_by_id(offer_id):
    offers_service_host, offers_service_port = get_offers_service()
    url = f'http://{offers_service_host}:{offers_service_port}/offers/{offer_id}'
    logger.debug(f"Asking for offer info from {url}")
    offers_response = requests.get(url)  # we let connection error propagate
    logger.debug(f"Got offer info: {offers_response}")
    if not offers_response.ok:
        logger.error(f"Offers service not giving us offers: {offers_response.reason}")
        raise BadRequest(message='Cannot obtain offers', status_code=500)
    offer = offers_response.json()['task']
    return offer


@app.route('/coupon_offer', methods=['GET'])
def coupon_offer():
    """
    Returns an offer recommendation for a given user.

    Hits the offers endpoint to find what offers are available, get their preferences for adjusting scores.
    Uses Amazon Personalize if available to score them.
    Returns the highest scoring offer.

    Experimentation is disabled because we are sending the offers through Pinpoint emails and for this
    demonstration we would need to add some more complexity to track those within the current framework.
    Pinpoint also has A/B experimentation built in which can be used.
    """

    user_id = request.args.get('userID')
    if not user_id:
        raise BadRequest('userID is required')

    resp_headers = {}
    try:

        inference_arn = get_parameter_values(offers_arn_param_name)[0]
        offers_service_host, offers_service_port = get_offers_service()

        url = f'http://{offers_service_host}:{offers_service_port}/offers'
        app.logger.debug(f"Asking for offers info from {url}")
        offers_response = requests.get(url)
        app.logger.debug(f"Got offer info: {offers_response}")

        if not offers_response.ok:
            app.logger.exception('Offers service did not return happily', offers_response.reason())
            raise BadRequest(message='Cannot obtain offers', status_code=500)
        else:

            offers = offers_response.json()['tasks']
            offers_by_id = {str(offer['id']): offer for offer in offers}
            offer_ids = sorted(list(offers_by_id.keys()))

            if not inference_arn:
                app.logger.warning('No campaign Arn set for offers - returning arbitrary')
                # We deterministically choose an offer
                # - random approach would have been chosen_offer_id = random.choice(offer_ids)
                chosen_offer_id = offer_ids[int(user_id) % len(offer_ids)]
                chosen_score = None
            else:
                resp_headers['X-Personalize-Recipe'] = get_recipe(inference_arn)
                logger.info(f"Input to Personalized Ranking for offers: userId: {user_id}({type(user_id)}) "
                            f"inputList: {offer_ids}")
                get_recommendations_response = personalize_runtime.get_recommendations(
                    campaignArn=inference_arn,
                    userId=user_id,
                    numResults=len(offer_ids)
                )

                logger.info(f'Recommendations returned: {json.dumps(get_recommendations_response)}')

                return_key = 'itemList'
                # Here is where might want to incorporate some business logic
                # for more information on how these scores are used see
                # https://aws.amazon.com/blogs/machine-learning/introducing-recommendation-scores-in-amazon-personalize/

                # An alternative approach would be to train Personalize to produce recommendations based on objectives
                # we specify rather than the default which is to maximise the target event. For more information, see
                # https://docs.aws.amazon.com/personalize/latest/dg/optimizing-solution-for-objective.html

                user_scores = {item['itemId']: float(item['score']) for item in
                               get_recommendations_response[return_key]}
                # We assume we have pre-calculated the adjusting factor, can be a mix of probability when applicable,
                # calculation of expected return per offer, etc.
                adjusted_scores = {offer_id: score * offers_by_id[offer_id]['preference']
                                   for offer_id, score in user_scores.items()}
                logger.info(f"Scores after adjusting for preference parameters: {adjusted_scores}")

                # Normalise these - makes it easier to do further adjustments
                score_sum = sum(adjusted_scores.values())
                adjusted_scores = {offer_id: score / score_sum
                                   for offer_id, score in adjusted_scores.items()}
                logger.info(f"Scores after normalising adjusted scores: {adjusted_scores}")

                # Just one way we could add some randomness - adds serendipity though removes personalization a bit
                # because we have the scores though we retain a lot of the personalization
                random_factor = 0.0
                adjusted_scores = {offer_id: score*(1-random_factor) + random_factor*random.random()
                                   for offer_id, score in adjusted_scores.items()}
                logger.info(f"Scores after adding randomness: {adjusted_scores}")

                # We can do many other things here, like randomisation, normalisation in different dimensions,
                # tracking per-user, offer, quotas, etc. Here, we just select the most promising adjusted score
                chosen_offer_id = max(adjusted_scores, key=adjusted_scores.get)
                chosen_score = user_scores[chosen_offer_id]
                chosen_adjusted_score = adjusted_scores[chosen_offer_id]

            chosen_offer = offers_by_id[chosen_offer_id]
            if chosen_score is not None:
                chosen_offer['score'] = chosen_score
                chosen_offer['adjusted_score'] = chosen_adjusted_score

        resp = Response(json.dumps({'offer': chosen_offer}, cls=CompatEncoder),
                        content_type='application/json', headers=resp_headers)

        app.logger.debug(f"Recommendations response to be returned for offers: {resp}")
        return resp

    except Exception as e:
        app.logger.exception('Unexpected error generating recommendations', e)
        raise BadRequest(message='Unhandled error', status_code=500)


@app.route('/experiment/outcome', methods=['POST'])
def experiment_outcome():
    """ Tracks an outcome/conversion for an experiment """
    if request.content_type.startswith('application/json'):
        content = request.json
        app.logger.info(content)

        correlation_id = content.get('correlationId')
    else:
        correlation_id = request.form.get('correlationId')

    if not correlation_id:
        raise BadRequest('correlationId is required')

    exp_manager = ExperimentManager()

    try:
        experiment = exp_manager.get_by_correlation_id(correlation_id)
        if not experiment:
            return jsonify({ 'status_code': 404, 'message': 'Experiment not found' }), 404

        experiment.track_conversion(correlation_id, get_timestamp_from_request())

        return jsonify(success=True)

    except Exception as e:
        app.logger.exception('Unexpected error logging outcome', e)
        raise BadRequest(message='Unhandled error', status_code=500)

if __name__ == '__main__':

    if DEBUG_LOGGING:
        level = logging.DEBUG
    else:
        level = logging.INFO
    app.logger.setLevel(level)
    if EXPERIMENTATION_LOGGING:
        logging.getLogger('experimentation').setLevel(level=level)
        logging.getLogger('experimentation.experiment_manager').setLevel(level=level)
        for handler in app.logger.handlers:
            logging.getLogger('experimentation').addHandler(handler)
            logging.getLogger('experimentation.experiment_manager').addHandler(handler)
            handler.setLevel(level)  # this will get the main app logs to CloudWatch

    app.wsgi_app = LoggingMiddleware(app.wsgi_app)

    app.run(debug=True, host='0.0.0.0', port=80)


INDEX_DOES_NOT_EXIST = 'index_not_found_exception'

search_domain_scheme = os.environ.get('OPENSEARCH_DOMAIN_SCHEME', 'https')
search_domain_host = os.environ['OPENSEARCH_DOMAIN_HOST']
search_domain_port = os.environ.get('OPENSEARCH_DOMAIN_PORT', 443)
INDEX_PRODUCTS = 'products'

search_client = OpenSearch(
    [search_domain_host],
    scheme=search_domain_scheme,
    port=search_domain_port,
)

# -- Logging
class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)

# -- End Logging

app = Flask(__name__)
corps = CORS(app)

xray_recorder.configure(service='Search Service')
XRayMiddleware(app, xray_recorder)

# -- Exceptions
class BadRequest(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

# -- Utilities
def get_offset_and_size(request):
    offset = request.args.get('offset', default = 0, type = int)
    if offset < 0:
        raise BadRequest('offset must be greater than or equal to zero')
    size = request.args.get('size', default = 10, type = int)
    if size < 1:
        raise BadRequest('size must be greater than zero')

    return offset, size

# -- Handlers

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/')
def index():
    return 'Search Service'

@app.route('/search/products', methods=['GET'])
def search_products():
    search_term = request.args.get('searchTerm')
    if not search_term:
        raise BadRequest('searchTerm is required')
    search_term = search_term.lower()

    offset, size = get_offset_and_size(request)
    collapse_size = int(max(size / 15, 15))
    app.logger.info('Searching products for "%s" starting at %d and returning %d hits with collapse size of %d',
        search_term, offset, size, collapse_size
    )

    try:
        # Query OpenSearch using a disjunction max query across multiple fields using the "match_bool_prefix".
        # The "match_bool_prefix" query will tokenize the search expression where the last token is turned into
        # a prefix query. This is good match for an "auto-complete" search UX.
        # To improve the diversity of hits across categories (particularly important when the search expression is
        # short/vague), the search is collapsed on the category keyword field. This ensures that the top hits are pulled
        # from all categories which are then aggregated into a unified response.
        results = search_client.search(index = INDEX_PRODUCTS, body={
            "from": offset,
            "size": size,
            "query": {
                "dis_max" : {
                    "queries" : [
                        { "match_bool_prefix" : { "name" : { "query": search_term, "boost": 1.2 }}},
                        { "match_bool_prefix" : { "category" : search_term }},
                        { "match_bool_prefix" : { "style" : search_term }},
                        { "match_bool_prefix" : { "description" : { "query": search_term, "boost": 0.6 }}}
                    ],
                    "tie_breaker" : 0.7
                }
            },
            "fields":[
                "_id"
            ],
            "_source": False,
            "collapse": {
                "field": "category.keyword",
                "inner_hits": {
                    "name": "category_hits",
                    "size": collapse_size,
                    "fields":[
                        "_id"
                    ],
                    "_source": False
                }
            }
        })

        app.logger.debug(json.dumps(results))

        # Because we're collapsing results across categories, the total hits will likely be > size.
        total_hits = results["hits"]["total"]["value"]
        app.logger.debug('Total hits across categories: %d', total_hits)

        cats_with_hits = len(results["hits"]["hits"])
        avg_hits_cat = int(size / cats_with_hits) if cats_with_hits > 0 else 0
        app.logger.debug('Average hits per category: %d', avg_hits_cat)
        hits_for_cats = []
        accum_hits = 0
        cats_with_more = 0

        # Determine the number of hits per category that we can use.
        for item in results['hits']['hits']:
            cat_hits = item["inner_hits"]["category_hits"]["hits"]["total"]["value"]
            if cat_hits > avg_hits_cat:
                cats_with_more += 1
            hits_this_cat = min(cat_hits, avg_hits_cat)
            accum_hits += hits_this_cat
            hits_for_cats.append([cat_hits, hits_this_cat])

        if accum_hits < size and cats_with_more:
            # Still more room available. Add more items across categories that have more than average.
            more_each = int((size - accum_hits) / cats_with_more)
            for counts in hits_for_cats:
                more_this_cat = min(more_each, counts[0] - counts[1])
                accum_hits += more_this_cat
                counts[1] += more_this_cat

        found_items = []

        for idx, item in enumerate(results['hits']['hits']):
            cat_hits = item["inner_hits"]["category_hits"]["hits"]["hits"]

            if accum_hits < size and hits_for_cats[idx][1] < hits_for_cats[idx][0]:
                # If still more room available, use first one with more to give.
                to_add = min(size - accum_hits, hits_for_cats[idx][0] - hits_for_cats[idx][1])
                hits_for_cats[idx][1] += to_add
                accum_hits += to_add

            added = 0
            for hit in cat_hits:
                found_items.append({
                    'itemId': hit['_id']
                })
                added += 1
                if added == hits_for_cats[idx][1]:
                    break

        return json.dumps(found_items)

    except NotFoundError as e:
        if e.error == INDEX_DOES_NOT_EXIST:
            app.logger.error('Search index does not exist')
            raise BadRequest(message = 'Index does not exist yet; please complete search workshop', status_code = 404)
        raise BadRequest(message = 'Not Found', status_code = 404)

    except Exception as e:
        app.logger.exception('Unexpected error performing product search', e)
        raise BadRequest(message = 'Unhandled error', status_code = 500)

@app.route('/similar/products', methods=['GET'])
def similar_products():
    product_id = request.args.get('productId')
    if not product_id:
        raise BadRequest('productId is required')
    offset, size = get_offset_and_size(request)
    app.logger.info(f'Searching for similar products to "{product_id}" starting at {offset} and returning {size} hits')

    try:
        results = search_client.search(index = INDEX_PRODUCTS, body={
            "from": offset,
            "size": size,
                "query": {
                    "more_like_this": {
                        "fields": ["name", "category", "style", "description"],
                        "like": [{
                            "_index": INDEX_PRODUCTS,
                            "_id": product_id
                        }],
                        "min_term_freq" : 1,
                        "max_query_terms" : 10
                    }
                }
            })

        app.logger.debug(json.dumps(results))

        found_items = []

        for item in results['hits']['hits']:
            found_items.append({
                'itemId': item['_id']
            })
        return json.dumps(found_items)

    except NotFoundError as e:
        if e.error == INDEX_DOES_NOT_EXIST:
            app.logger.error('Search index does not exist')
            raise BadRequest(message = 'Index does not exist yet; please complete search workshop', status_code = 404)
        raise BadRequest(message = 'Not Found', status_code = 404)

    except Exception as e:
        app.logger.exception('Unexpected error performing similar product search', e)
        raise BadRequest(message = 'Unhandled error', status_code = 500)

if __name__ == '__main__':
    app.wsgi_app = LoggingMiddleware(app.wsgi_app)
    app.run(debug=True,host='0.0.0.0', port=80)


# -- Environment variables - defined by CloudFormation when deployed
VIDEO_BUCKET = os.environ.get('RESOURCE_BUCKET')
IMAGE_ROOT_URL = os.environ.get('IMAGE_ROOT_URL') + 'videos/'
SSM_VIDEO_CHANNEL_MAP_PARAM = os.environ.get('PARAMETER_IVS_VIDEO_CHANNEL_MAP', 'retaildemostore-ivs-video-channel-map')

USE_DEFAULT_IVS_STREAMS = os.environ.get('USE_DEFAULT_IVS_STREAMS') == 'true'

DEFAULT_THUMB_FNAME = 'default_thumb.png'
STATIC_FOLDER = '/app/static'
STATIC_URL_PATH = '/static'
SUBTITLE_FORMAT = 'srt'
LOCAL_VIDEO_DIR = '/app/video-files/'
DEFAULT_STREAMS_CONFIG_S3_PATH = 'videos/default_streams/default_streams.json'

# -- Parameterised ffmpeg commands
FFMPEG_STREAM_CMD = """ffmpeg -loglevel panic -hide_banner -re -stream_loop -1 -i \"{video_filepath}\" \
                           -r 30 -c:v copy -f flv rtmps://{ingest_endpoint}:443/app/{stream_key} -map 0:s -f {subtitle_format} -"""
FFMPEG_SUBS_COMMAND = "ffmpeg -i \"{video_filepath}\" \"{subtitle_path}\""


# Globally accessed variable to store stream metadata (URLs & associated product IDs). Returned via `stream_details`
# endpoint
stream_details = {}

ivs_client = boto3.client('ivs')
ssm_client = boto3.client('ssm')
s3_client = boto3.client('s3')


# -- Load default streams config
def load_default_streams_config():
    app.logger.info(f"Downloading default streams config from from bucket {VIDEO_BUCKET} with key {DEFAULT_STREAMS_CONFIG_S3_PATH}.")

    config_response = s3_client.get_object(Bucket=VIDEO_BUCKET, Key=DEFAULT_STREAMS_CONFIG_S3_PATH)
    config = json.loads(config_response['Body'].read().decode('utf-8'))
    for (key, entry) in config.items():
        app.logger.info(f"{key}, {entry}")
        config[key] = {**entry, 'thumb_url': IMAGE_ROOT_URL + entry['thumb_fname']}
        config[key].pop('thumb_fname', None)

    app.logger.info("Pulled config:")
    app.logger.info(config)

    return config


# -- Video streaming
def download_video_file(s3_key):
    """
        Downloads a video file and associated thumbnail from S3. Thumbnails are identified by a .png file with the same
        name and in the same location as the video.
    """
    local_path = LOCAL_VIDEO_DIR + s3_key.split('/')[-1]
    app.logger.info(f"Downloading file {s3_key} from bucket {VIDEO_BUCKET} to {local_path}.")
    s3_client.download_file(Bucket=VIDEO_BUCKET, Key=s3_key, Filename=local_path)
    app.logger.info(f"File {s3_key} downloaded from bucket {VIDEO_BUCKET} to {local_path}.")

    thumbnail_path = None
    thumbnail_key = '.'.join(s3_key.split('.')[:-1]) + '.png'
    try:
        local_thumbnail_fname = thumbnail_key.split('/')[-1]
        thumbnail_path = IMAGE_ROOT_URL + local_thumbnail_fname
    except Exception as e:
        app.logger.warning(f'No thumbnail available for {VIDEO_BUCKET}/{s3_key} as {VIDEO_BUCKET}/{thumbnail_key} - '
                           f'exception: {e}')
    return local_path, thumbnail_path


def get_ffmpeg_stream_cmd(video_filepath, ingest_endpoint, stream_key, subtitle_format):
    """
        Returns the command to start streaming a video using ffmpeg.
    """
    return FFMPEG_STREAM_CMD.format(**locals())


def get_ffmpeg_subs_cmd(video_filepath, subtitle_path):
    """
        Returns the ffmpeg command to rip subtitles (ie. metadata) from a video file.
    """
    return FFMPEG_SUBS_COMMAND.format(**locals())


def get_featured_products(video_filepath, channel_id):
    """
        Extracts a list of product IDs from the metadata attached to a video file. The values are saved in the global
        `stream_details` dict.
    """
    subtitle_path = pathlib.Path(video_filepath).with_suffix('.srt')
    get_subs_command = get_ffmpeg_subs_cmd(video_filepath, subtitle_path)
    process = subprocess.run(
                    get_subs_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
    with open(subtitle_path) as f:
        subtitle_content = srt.parse(f)
        for line in subtitle_content:
            product_id = json.loads(line.content)['productId']
            if 'products' not in stream_details[channel_id]:
                stream_details[channel_id]['products'] = [product_id]
            else:
                if product_id not in stream_details[channel_id]['products']:
                    stream_details[channel_id]['products'].append(product_id)


def is_ssm_parameter_set(parameter_name):
    """
        Returns whether an SSM parameter with a given name has been set (ie. value is not 'NONE')
    """
    try:
        response = ssm_client.get_parameter(Name=parameter_name)
        return response['Parameter']['Value'] != 'NONE'
    except ssm_client.exceptions.ParameterNotFound:
        return False


def log_ffmpeg_processes():
    app.logger.info('Running ffmpeg processes:')
    app.logger.info(os.system("ps aux|grep 'PID\|ffmpeg'"))


def put_ivs_metadata(channel_arn, line):
    """
        Sends metadata to a given IVS stream. Metadata can be any string, but the AWS Retail Demo Store UI expects
        metadata of the format {"productId":"<product-id>"}
    """
    try:
        app.logger.info(f'Sending metadata to stream: {line}')
        ivs_client.put_metadata(
            channelArn=channel_arn,
            metadata=line
        )
    except ivs_client.exceptions.ChannelNotBroadcasting as ex:
        app.logger.warning(f'Channel not broadcasting. Waiting for 5 seconds. Exception: {ex}')
        log_ffmpeg_processes()
        time.sleep(5)
    except ivs_client.exceptions.InternalServerException as ex:
        app.logger.error(f'We have an internal error exception. Waiting for 30 seconds. Exception: {ex}')
        log_ffmpeg_processes()
        time.sleep(30)


def get_stream_state(channel_arn):
    """
        Returns the state of a stream given it's ARN. One of 'LIVE', 'OFFLINE' (from API response)
        or 'NOT_BROADCASTING' (inferred).
    """
    try:
        stream_response = ivs_client.get_stream(channelArn=channel_arn)['stream']
        stream_state = stream_response['state']
    except ivs_client.exceptions.ChannelNotBroadcasting:
        stream_state = "NOT_BROADCASTING"
    return stream_state


def start_streams():
    """
        Initiates all IVS streams based on environment variables. If the SSM_VIDEO_CHANNEL_MAP_PARAM (map of videos in
        S3 to IVS channels) is set and the user has not requested to use the default IVS streams
        (USE_DEFAULT_IVS_STREAMS, defined by CloudFormation input) then one stream will be started per video described
        in the video to IVS channel map. Each stream runs in a separate thread.

        If streams are not started, then `stream_details` will be set to the details of a collection of existing streams
    """
    if is_ssm_parameter_set(SSM_VIDEO_CHANNEL_MAP_PARAM) and not USE_DEFAULT_IVS_STREAMS:
        video_channel_param_value = ssm_client.get_parameter(Name=SSM_VIDEO_CHANNEL_MAP_PARAM)['Parameter']['Value']
        app.logger.info(f"Found IVS channel map: {video_channel_param_value}")
        video_channel_map = json.loads(video_channel_param_value)

        for idx, (s3_video_key, ivs_channel_arn) in enumerate(video_channel_map.items()):
            threading.Thread(target=stream, args=(s3_video_key, ivs_channel_arn, idx)).start()

    else:
        global stream_details
        stream_details = load_default_streams_config()


def stream(s3_video_key, ivs_channel_arn, channel_id):
    """
        Starts the stream for a given video file and IVS channel. The video file is streamed on a loop using ffmpeg, and
        any attached metadata (from the subtitles embedded in the video file) is sent to the channel's `put_metadata`
        endpoint.
    """
    video_filepath, thumb_url = download_video_file(s3_video_key)
    if thumb_url is None:
        thumb_url = IMAGE_ROOT_URL + DEFAULT_THUMB_FNAME

    channel_response = ivs_client.get_channel(arn=ivs_channel_arn)['channel']
    ingest_endpoint = channel_response['ingestEndpoint']
    playback_endpoint = channel_response['playbackUrl']
    stream_details[channel_id] = {'playback_url': playback_endpoint,
                                  'thumb_url': thumb_url}

    get_featured_products(video_filepath, channel_id)

    stream_state = get_stream_state(ivs_channel_arn)
    stream_arn = ivs_client.list_stream_keys(channelArn=ivs_channel_arn)['streamKeys'][0]['arn']
    stream_key = ivs_client.get_stream_key(arn=stream_arn)['streamKey']['value']
    app.logger.info(f"Stream details:\nIngest endpoint: {ingest_endpoint}\nStream state: {stream_state}")

    if SUBTITLE_FORMAT == 'srt':
        while True:
            if stream_state != "NOT_BROADCASTING":
                app.logger.info(f"Stream {stream_arn} is currently in state {stream_state}. Waiting for state NOT_BROADCASTING")
                sleep_time = 20
                app.logger.info(f"Waiting for {sleep_time} seconds")
                time.sleep(sleep_time)
                stream_state = get_stream_state(ivs_channel_arn)
                continue

            app.logger.info('Starting video stream')
            ffmpeg_stream_cmd = get_ffmpeg_stream_cmd(video_filepath, ingest_endpoint, stream_key, SUBTITLE_FORMAT)
            app.logger.info(f'ffmpeg command: {ffmpeg_stream_cmd}')

            process = subprocess.Popen(
                ffmpeg_stream_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True)
            app.logger.info('Running ffmpeg processes:')
            app.logger.info(os.system("ps aux|grep 'PID\|ffmpeg'"))

            lines = iter(process.stdout)
            app.logger.info('Starting event stream')
            while True:
                try:
                    int(next(lines).strip())
                    time_range = next(lines).strip()
                    if not '-->' in time_range:
                        raise ValueError(f'Expected a time range instead of {time_range}')
                    send_text = ''
                    while True:
                        text = next(lines).strip()
                        if len(text) == 0: break
                        if len(send_text)>0: send_text+='\n'
                        send_text += text
                    put_ivs_metadata(ivs_channel_arn, send_text)
                except StopIteration:
                    app.logger.warning('Video iteration has stopped unexpectedly. Attempting restart in 10 seconds.')
                    time.sleep(10)
                    break
    else:
        raise NotImplementedError(f'{SUBTITLE_FORMAT} is not currently supported by this demo.')
# -- End Video streaming


# -- Logging
class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)
# -- End Logging


# -- Exceptions
class BadRequest(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


# -- Handlers
app = Flask(__name__,
            static_folder=STATIC_FOLDER,
            static_url_path=STATIC_URL_PATH)
corps = CORS(app)


xray_recorder.configure(service='Videos Service')
XRayMiddleware(app, xray_recorder)

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
def index():
    return 'Videos Service'


@app.route('/stream_details')
def streams():
    response_data = []
    for value in stream_details.values():
        response_data.append(value)
    response = {
        "streams": response_data
    }
    return Response(json.dumps(response), content_type = 'application/json')


@app.route('/health')
def health():
    return 'OK'


if __name__ == '__main__':
    app.wsgi_app = LoggingMiddleware(app.wsgi_app)
    app.logger.setLevel(level=logging.INFO)

    app.logger.info(f"VIDEO_BUCKET: {VIDEO_BUCKET}")
    app.logger.info(f"SSM_VIDEO_CHANNEL_MAP_PARAM: {SSM_VIDEO_CHANNEL_MAP_PARAM}")
    app.logger.info(f"USE_DEFAULT_IVS_STREAMS: {USE_DEFAULT_IVS_STREAMS}")

    app.logger.info("Starting video streams")
    start_streams()

    app.logger.info("Starting API")
    app.run(debug=False, host='0.0.0.0', port=80)
