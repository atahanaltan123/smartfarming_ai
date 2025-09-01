import json
import os
import sys
from flask import Flask, request, jsonify, render_template_string

# Flask uygulamasını import et
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from app import app

def handler(event, context):
    """Netlify serverless function handler"""
    
    # HTTP method ve path'i al
    http_method = event.get('httpMethod', 'GET')
    path = event.get('path', '/')
    
    # Query parameters
    query_params = event.get('queryStringParameters') or {}
    
    # Headers
    headers = event.get('headers') or {}
    
    # Body
    body = event.get('body', '')
    
    # Flask test client oluştur
    with app.test_client() as client:
        # Request context oluştur
        with app.test_request_context(
            path=path,
            method=http_method,
            query_string=query_params,
            headers=headers,
            data=body
        ):
            try:
                # Flask route'u çalıştır
                response = client.open(
                    path=path,
                    method=http_method,
                    query_string=query_params,
                    headers=headers,
                    data=body
                )
                
                return {
                    'statusCode': response.status_code,
                    'headers': {
                        'Content-Type': response.content_type,
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Headers': 'Content-Type',
                        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
                    },
                    'body': response.get_data(as_text=True)
                }
                
            except Exception as e:
                return {
                    'statusCode': 500,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'error': f'Server error: {str(e)}'
                    })
                }
