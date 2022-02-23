# Instructions

---

## Configuration:

##### [docker-compose.yaml](docker-compose.yaml)

    service -> database -> environment:
        POSTGRES_USER - Name of user to create
        POSTGRES_PASSWORD - Password for created user
        POSTGRES_DB - Name of database to create

    service -> app -> environment:
        PG_USER - Username to use for connection to database
        PG_PASSWORD - Password to user for connection to database
        PG_DB - Name of database to connect
        PG_HOST - Host of database to connect
        PG_PORT - Port of datase to connect

##### [app/config.py](app/config.py)

    TABLE_NAME - Name of table in database to create
    TIME_RECORD_PATH - Filepath to file with time records
    RESULT_PLOT_PATH - File to file with image of result plot

    RESULT_CSV_PATH - Filepath to CSV file with results data
    RESULT_CSV_ENCODING - Encoding for results CSV file
    RESULT_CSV_DELIMITER - Delimiter for results CSV file

---

## Start the application:

1. Start `Docker Engine`.


2. Download/unpack project into desire folder.  


3. Open console in the same folder as project.  


4. Run following command to build application:
    ```
    $ docker-compose build
    ```

5. Run following command to *start* application:
   ```
   $ docker-compose up
   ```

**Note**: Command to *stop* application:
```
$ docker-compose down
```

---

## Simulation of application failure:

1. *Start* the application.


2. Try to kill/stop one of existing containers:
```
$ docker kill [CONTAINER_NAME/CONTAINER_ID]
```

4. Then after a while start killed containers:
```
$ docker start [CONTAINER_NAME/CONTAINER_ID]
```

**Note**: *After a while, the program should continue to run normally*

---