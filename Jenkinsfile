pipeline {
    agent any
    environment {
        DOCKER_HUB_USER = credentials('dockerhub-user')
        DOCKER_HUB_PASS = credentials('dockerhub-pass')
    }
    stages {
        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $DOCKER_HUB_USER/garbage-classifier:latest .'
            }
        }
        stage('Push to DockerHub') {
            steps {
                sh 'echo $DOCKER_HUB_PASS | docker login -u $DOCKER_HUB_USER --password-stdin'
                sh 'docker push $DOCKER_HUB_USER/garbage-classifier:latest'
            }
        }
    }
}
