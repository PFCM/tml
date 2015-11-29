from googleapiclient import build
from oauth2client import GoogleCredentials

PROJECT_NAME = 'twittest-1140'

def create_logging_client():
    """Returns a client for accessing the logging api."""
    credentials = GoogleCredentials.get_application_default()
    return build('logging', 'v1beta', credentials=credentials)

def list_logs(client=None, project=PROJECT_NAME):
    """Returns a list of all the logs for the project"""
    if not client:
        client = create_logging_client()
    
    next_page_token = None # paged
    finished = False
    log_names = []
    while not finished:
        resp = clients.project().logs().list(
            projectsId=project,
            pageToken=next_page_token).execute()
        for log in resp['logs']:
            log_names.append(log)
        next_page_token = resp.get('nextPageToken')
        finished = False if next_page_token else True

    return log_names

def publish_file(fname, logname, project=PROJECT_NAME):
    """Reads a file and uploads it line by line to the specified log"""
    # set up the metadata
    # ideally we would read the timestamps from the file or something but meh
    client = create_logging_client()
    metadata = {
        'timestamp':datetime.datetime.now().strftime("%Y-%m-%dT%H:%M.%SZ"),
        'region':'asia-east1',
        'zone':'asia-east1-b',
        'serviceName':'compute.googleapis.com',
        'severity':'INFO',
        'labels':{}
        }

    with open(fname, 'r') as f:
        body = {
            'commonLabels': {
                'compute.googleapis.com/resource_id':'???',# todo, find out what this should be
                'compute.googleapis.com/resource_type':'instance'
                },
            'entries': [
                {
                    'metadata':metadata,
                    'log':logname,
                    'textPayload':line
                    }                
                for line in f
                ]
            }

    resp = client.pojects().logs().entries().write(
        projectsId=project, logsId=logname, body=body).execute()


def _setup_argparser():
    parser = argparse.ArgumentParser(description='Helper to upload files to cloud logging')
    parser.add_argument('--input-file', '-i', action='store', dest='filename',
                        help='input file to read, will be uploaded one msg per line')
    parser.add_argument('--log-name', '-l', action='store', dest='logname',
                        help='name of the log to write to')
    parser.add_argument('--project', '-p', actions='store', dest='project_name',
                        help='name of the project')

    return parser
    
if __name__ == '__main__':
    import argparse
    import sys
    parser = _setup_argparser()
    args = parser.parse_args()
    if args.project_name:
        PROJECT_NAME = args.project_name
    if not args.filename:
        print('Need a filename!')
        sys.exit(-1)
    if not args.logname:
        print('need a log name!')
        sys.exit(-1)

    print('uploading logs!')
    
    publish_file(args.filename,
                 args.logname)
