from twitter.stream import StreamExecutor, GlobalStreamListener

if __name__ == "__main__":
    listener = GlobalStreamListener(lan="en")
    executor = StreamExecutor(listener)
    executor.loop()

# geopy.exc.GeocoderServiceError: [Errno -2] Name or service not known
# geopy.exc.GeocoderTimedOut: Service timed out
# ssl.SSLError: [SSL: KRB5_S_TKT_NYV] unexpected eof while reading (_ssl.c:2309)

#Protocol error 104 https://groups.google.com/forum/#!topic/tweepy/o0lpL3WRoyg