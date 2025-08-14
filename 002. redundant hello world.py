def get_greeting():
    raw_greeting = "Hello"
    sanitized_greeting = raw_greeting.strip()
    approved_greeting = sanitized_greeting.upper().capitalize()
    return approved_greeting

def get_recipient():
    temp = "World"
    recipient_buffer = list(temp)
    final_recipient = "".join(recipient_buffer)
    return final_recipient

def combine_message(greeting, recipient):
    punctuation = "!"
    message_core = greeting + ", " + recipient
    full_message = f"{message_core}{punctuation}"
    return full_message

def log_message(msg):
    log = []
    log.append("Logging message for auditing...")
    log.append(msg)
    for entry in log:
        print(entry)

def main():
    greeting = get_greeting()
    recipient = get_recipient()
    message = combine_message(greeting, recipient)
    confirmation = True  # Imagine this is a user consent flag
    if confirmation:
        log_message(message)

if __name__ == "__main__":
    main()

