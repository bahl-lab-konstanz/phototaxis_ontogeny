
# Imports
import datetime
import getpass
import os
import warnings
import keyring

# Third party library imports
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


def store_slack_token(slack_token: str, bot_name="bahl-lab_bot", ):
    slack_token = str(slack_token).replace("'", "").replace('"', '')  # Remove extra ' and " from string
    keyring.set_password("slack_token", bot_name, slack_token)


def send_slack_message(to: str or list, message: str, bot_name="bahl-lab_bot", add_meta_data=True):
    if isinstance(to, type(None)):
        return

    # Add meta data to message
    if add_meta_data:
        now = datetime.datetime.now()
        now_str = f'Time: {now.strftime("%Y-%m-%d %H:%M:%S")}'
        if 'SETUP_INDEX' in os.environ.keys():
            id = f"Setup {os.environ['SETUP_INDEX']}"
        elif 'USER' in os.environ.keys():
            id = f"USer: {os.environ['USER']}"
        else:
            id = 'unknown'
        message = f"{id} | {now_str}\n{message}"

    # Retrieve token for bot
    slack_token = keyring.get_password("slack_token", bot_name)
    if isinstance(slack_token, type(None)):
        slack_token = getpass.getpass(f"Please specify slack_token for {bot_name}:")
        store_slack_token(bot_name, slack_token)

    # Connect to client
    client = WebClient(token=slack_token)

    # Prepare receivers
    try:
        ul = client.users_list()
    except Exception as e:
        warnings.warn(f'Slack message not send to {to}\n'
                      f'\tError:   {e}')
        return

    all_members = [user["profile"]['real_name'] for user in ul.data["members"]]
    if isinstance(to, str):
        receivers = [to]
    else:
        receivers = to

    # Loop over receivers and send message
    for receiver in receivers:
        if receiver not in all_members:
            warnings.warn(f'Receiver not found: {receiver}')
            continue
        else:
            user_index = all_members.index(receiver)
            user = ul.data["members"][user_index]
            chat_id = user["id"]
            try:
                response = client.chat_postMessage(
                    channel=chat_id,
                    text=message
                )
            except SlackApiError as e:
                warnings.warn(f'Slack message not send to {receiver}\n'
                              f'\tError:   {e}')


if __name__ == '__main__':
    # Store slack tocken
    # store_slack_tocken()

    # Try to send a slack message
    send_slack_message(
        to="Max Capelle",
        message="This is a test message from the slack_helper.py"
    )

